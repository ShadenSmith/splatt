
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "ttm.h"
#include "thd_info.h"
#include "tile.h"
#include "util.h"

#include <omp.h>

#define TTM_TILED 1



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

int splatt_ttmc(
    splatt_idx_t const mode,
    splatt_idx_t const * const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const tenout,
    double const * const options)
{
  idx_t const nmodes = tensors->nmodes;

  idx_t maxcols = 0;

  /* fill matrix pointers  */
  matrix_t * mats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (matrix_t *) malloc(sizeof(matrix_t));
    mats[m]->I = tensors->dims[m];
    mats[m]->J = ncolumns[m],
    mats[m]->rowmajor = 1;
    mats[m]->vals = matrices[m];

    maxcols = SS_MAX(maxcols, ncolumns[m]);
  }

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  idx_t const nthreads = (idx_t) options[SPLATT_OPTION_NTHREADS];
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 1,
    (maxcols * sizeof(val_t)) + 64);

  ttmc_csf(tensors, mats, tenout, mode, thds, options);

  /* cleanup */
  thd_free(thds, nthreads);
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]);
  }

  return SPLATT_SUCCESS;
}




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Compute (rowA^T rowB) and accumulate into row-major 'out'.
*
* @param rowA The first row vector.
* @param nA The number of elements in rowA.
* @param rowB The second row vector.
* @param nB The number of elements in rowB.
* @param[out] out The output matrix which is (nA x nB).
*/
static inline void p_twovec_outer_prod(
    val_t const * const restrict rowA,
    idx_t const nA,
    val_t const * const restrict rowB,
    idx_t const nB,
    val_t * const restrict out)
{
  for(idx_t i=0; i < nA; ++i) {
    val_t * const restrict orow = out + (i*nB);
    val_t const ival = rowA[i];
    for(idx_t j=0; j < nB; ++j) {
      orow[j] += ival * rowB[j];
    }
  }
}


/**
* @brief Compute C = A * B^T, where A is a column-major matrix of the gathered
*        rows related to fiber ids. B is a column-major matrix of the
*        accumulated nonzero values.
*
* @param fids_buf The gathered rows of the factor matrix (from fiber ids).
* @param ncol_fids The number of columns in the gathered rows, or the number of
*                  rows in the column-major matrix.
* @param accums_buf The accumulated nonzero contributions.
* @param ncol_accums The number of columns in the accumulation.
* @param nfibers The number of total fibers (outer products) to work on.
* @param[out] out The output matrix, which is (ncol_fids x ncol_accums).
*/
static inline void p_twovec_outer_prod_tiled(
    val_t const * const restrict fids_buf,
    idx_t const ncol_fids,
    val_t const * const restrict accums_buf,
    idx_t const ncol_accums,
    idx_t const nfibers,
    val_t * const restrict out)
{
#ifdef SPLATT_USE_BLAS
  val_t alpha = 1.;
  val_t beta = 1.;
  char transA = 'N';
  char transB = 'T';
  int M = ncol_accums;
  int N = ncol_fids;
  int K = nfibers;

	BLAS_GEMM(
      &transA, &transB, /* transposes */
      &M, &N, &K,       /* dimensions */
      &alpha,
      accums_buf, &M,   /* A */
      fids_buf, &N,     /* B */
      &beta, out, &M);  /* C */

#else
  /* (possibly slow) summation of outer products */
  for(idx_t f=0; f < nfibers; ++f) {
    val_t const * const restrict rowA = fids_buf + (f * ncol_fids);
    val_t const * const restrict rowB = accums_buf + (f * ncol_accums);

    p_twovec_outer_prod(rowA, ncol_fids, rowB, ncol_accums, out);
  }
#endif
}


/**
* @brief Calculate the size of the output tensor and zero it.
*
* @param tenout The output tensor.
* @param mats The input matrices (they determine the size of the output).
* @param nmodes The number of modes.
* @param mode The mode we are computing.
* @param dims The dimensions of the input tensor.
*/
static inline void p_clear_tenout(
    val_t * const restrict tenout,
    matrix_t ** mats,
    idx_t const nmodes,
    idx_t const mode,
    idx_t const * const dims)
{
  /* clear tenout */
  idx_t outsize = dims[mode];
  for(idx_t m=0; m < nmodes; ++m) {
    if(m != mode) {
      outsize *= mats[m]->J;
    }
  }
  memset(tenout, 0, outsize * sizeof(*tenout));
}


/**
* @brief Perform TTM on the root mode of a CSF tensor. No locks are used.
*
* @param csf The input tensor.
* @param tile_id Which tile.
* @param mats Input matrices.
* @param tenout Output tensor.
* @param thds Thread structures.
*/
static void p_csf_ttmc_root3(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{
  assert(csf->nmodes == 3);

  matrix_t const * const A = mats[csf->dim_perm[1]];
  matrix_t const * const B = mats[csf->dim_perm[2]];

  idx_t const rankA = A->J;
  idx_t const rankB = B->J;

  val_t const * const restrict vals = csf->pt[tile_id].vals;

  idx_t const * const restrict sptr = csf->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = csf->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = csf->pt[tile_id].fids[0];
  idx_t const * const restrict fids = csf->pt[tile_id].fids[1];
  idx_t const * const restrict inds = csf->pt[tile_id].fids[2];

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  int const tid = omp_get_thread_num();
  val_t * const restrict accum_nnz_raw = (val_t *) thds[tid].scratch[0];

  /* tiled outer products */
  idx_t naccum;
  idx_t * const accum_fids  = thds[tid].scratch[1];
  val_t * const accum_oprod = thds[tid].scratch[2];

  /* foreach slice */
  idx_t const nslices = csf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    val_t * const restrict outv = tenout + (fid * rankA * rankB);

    naccum = 0;

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* grab nnz accumulation buffer */
      val_t * const restrict accum_nnz = accum_nnz_raw + (naccum * rankB);

      /* first entry of the fiber is used to initialize accum_nnz */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * rankB);
      for(idx_t r=0; r < rankB; ++r) {
        accum_nnz[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * rankB);
        for(idx_t r=0; r < rankB; ++r) {
          accum_nnz[r] += v * bv[r];
        }
      }

#if TTM_TILED
      /* accumulate outer product into tenout */
      accum_fids[naccum++] = fids[f];
#else
      val_t const * const restrict av = avals  + (fids[f] * rankA);
      p_twovec_outer_prod(av, rankA, accum_nnz, rankB, outv);
#endif
    } /* foreach fiber */

    /* OUTER PRODUCTS */

#if TTM_TILED
    /* gather rows into accum_oprod */
    for(idx_t r=0; r < naccum; ++r) {
      memcpy(accum_oprod + (r * rankA), avals + (accum_fids[r] * rankA),
          rankA * sizeof(*accum_oprod));
    }

    /* tiled outer product */
    p_twovec_outer_prod_tiled(accum_oprod, rankA, accum_nnz_raw, rankB, naccum,
        outv);
#endif

  } /* foreach outer slice */
}



static void p_csf_ttmc_root(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = csf->nmodes;
  val_t const * const vals = csf->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }
#if 1
  if(nmodes == 3) {
    p_csf_ttmc_root3(csf, tile_id, mats, tenout, thds);
    return;
  }
#endif

  idx_t const mode = csf->dim_perm[0];

  idx_t ncols = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    if(m != mode) {
      ncols *= mats[m]->J;
    }
  }

  idx_t ncols_lvl[MAX_NMODES];
  ncols_lvl[0] = ncols;
  for(idx_t m=1; m < nmodes; ++m) {
    ncols_lvl[m] = ncols_lvl[m-1] / mats[csf->dim_perm[m]]->J;
  }

  val_t * buf[MAX_NMODES];
  printf("ncols:");
  for(idx_t m=0; m < nmodes-1; ++m) {
    printf(" %lu", ncols_lvl[m]);
    buf[m] = calloc(ncols_lvl[m], sizeof(**buf));
  }
  printf("\n");

  idx_t const nfibs = csf->pt[tile_id].nfibs[0];

  idx_t const * const * const restrict fp
      = (idx_t const * const *) csf->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) csf->pt[tile_id].fids;

  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nfibs; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

  }

  /* cleanup */
  for(idx_t m=0; m < nmodes-1; ++m) {
    free(buf[m]);
  }
}



static inline void p_root_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);
    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      p_csf_ttmc_root(tensor, 0, mats, tenout, thds);
      break;

    /* XXX */
    default:
      fprintf(stderr, "SPLATT: TTM does not support tiling yet.\n");
      exit(1);
      break;
    }
    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */

}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


void ttmc_csf(
    splatt_csf const * const tensors,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  /* clear out stale results */
  p_clear_tenout(tenout, mats, tensors->nmodes, mode, tensors->dims);

  omp_set_num_threads(opts[SPLATT_OPTION_NTHREADS]);

  /* choose which TTM function to use */
  splatt_csf_type which = opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {
  case SPLATT_CSF_ALLMODE:
    p_root_decide(tensors+mode, mats, tenout, mode, thds, opts);
    break;

  default:
    fprintf(stderr, "SPLATT: only SPLATT_CSF_ALLMODE supported for TTM.\n");
    exit(1);
  }
}



void ttmc_stream(
    sptensor_t const * const tt,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    double const * const opts)
{
  /* clear out stale results */
  p_clear_tenout(tenout, mats, tt->nmodes, mode, tt->dims);

  omp_set_num_threads(opts[SPLATT_OPTION_NTHREADS]);

  val_t const * const restrict vals = tt->vals;
  val_t * mvals[MAX_NMODES];
  idx_t nfactors[MAX_NMODES];

  idx_t ncols = 1;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    nfactors[m] = mats[m]->J;
    mvals[m] = mats[m]->vals;

    if(m != mode) {
      ncols *= nfactors[m];
    }
  }

  idx_t const nmodes = tt->nmodes;

  for(idx_t n=0; n < tt->nnz; ++n) {
    val_t * const restrict outrow = tenout + (tt->ind[mode][n] * ncols);

    /* foreach entry in the output 'slice' */
    for(idx_t f=0; f < ncols; ++f) {

      val_t accum = vals[n];

      /* we will modify this */
      idx_t colnum = f;

      /* we will map f to its (m-1)-dimensional coordinate which gives us its
       * column ids for each mat[m].
       *
       * start from the last mode and work backwards */
      idx_t col_id = f;
      for(idx_t m=nmodes; m-- != 0; ) {
        if(m == mode) {
          continue;
        }

        /* compute column id for mats[m] and update colnum */
        col_id = colnum % nfactors[m];
        colnum /= nfactors[m];

        val_t const * const restrict inrow = mvals[m] +
            (tt->ind[m][n] * nfactors[m]);
        accum *= inrow[col_id];
      }

      /* now write accum to output */
      outrow[f] += accum;
    }
  }
}


void ttmc_largest_outer(
    splatt_csf const * const tensors,
    idx_t * const outer_sizes,
    double const * const opts)
{
  idx_t const ntensors = csf_ntensors(tensors, opts);
  idx_t const nmodes = tensors->nmodes;

  memset(outer_sizes, 0, nmodes * sizeof(*outer_sizes));

  for(idx_t t=0; t < ntensors; ++t) {
    splatt_csf const * const csf = &(tensors[t]);
    idx_t const ntiles = csf->ntiles;

    for(idx_t tile=0; tile < ntiles; ++tile) {
      /* don't count mode above nnz; they accumulate instead of oprod */
      for(idx_t m=0; m < nmodes-2; ++m) {
        idx_t const madj = csf->dim_perm[m];
        idx_t const * const fptr = csf->pt[tile].fptr[m];
        idx_t const nfibs = csf->pt[tile].nfibs[m];

        for(idx_t f=0; f < nfibs; ++f) {
          outer_sizes[madj] = SS_MAX(outer_sizes[madj], fptr[f+1] - fptr[f]);
        }
      }
    }
  }
}


idx_t tenout_dim(
    idx_t const nmodes,
    idx_t const * const nfactors,
    idx_t const * const dims)
{
  idx_t maxdim = 0;

  /* compute the size for each mode and maintain max */
  for(idx_t m=0; m < nmodes; ++m) {
    idx_t nrows = dims[m];
    idx_t ncols = 1;
    for(idx_t m2=0; m2 < nmodes; ++m2) {
      if(m == m2) {
        continue;
      }
      ncols *= nfactors[m2];
    }
    maxdim = SS_MAX(maxdim, nrows * ncols);
  }

  return maxdim;
}


