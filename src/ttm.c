
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

#define NLOCKS 1024
static omp_lock_t locks[NLOCKS];
static int locks_initialized = 0;
static void p_init_locks()
{
  if (!locks_initialized) {
    for(int i=0; i < NLOCKS; ++i) {
      omp_init_lock(locks+i);
    }
    locks_initialized = 1;
  }
}


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
* @brief Compute (rowA^T rowB) into row-major 'out'. Any former values in 'out'
*        are overwritten.
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
      orow[j] = ival * rowB[j];
    }
  }
}

/**
* @brief Compute (rowA^T rowB) AND ACCUMULATE into row-major 'out'.
*
* @param rowA The first row vector.
* @param nA The number of elements in rowA.
* @param rowB The second row vector.
* @param nB The number of elements in rowB.
* @param[out] out The output matrix which is (nA x nB).
*/
static inline void p_twovec_outer_prod_accum(
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

    p_twovec_outer_prod_accum(rowA, ncol_fids, rowB, ncol_accums, out);
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
* @brief Perform TTM on the root mode of a 3D CSF tensor. No locks are used.
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
      p_twovec_outer_prod_accum(av, rankA, accum_nnz, rankB, outv);
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


static void p_csf_ttmc_intl3(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{

  matrix_t const * const A = mats[csf->dim_perm[0]];
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
  val_t * const restrict accum_nnz = (val_t *) thds[tid].scratch[0];

  /* foreach slice */
  idx_t const nslices = csf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    val_t const * const restrict av = avals + (fid * rankA);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {

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

      /* accumulate outer product into tenout */
      val_t * const restrict outv = tenout + (fids[f] * rankA * rankB);
      omp_set_lock(locks + (fids[f] % NLOCKS));
      p_twovec_outer_prod_accum(av, rankA, accum_nnz, rankB, outv);
      omp_unset_lock(locks + (fids[f] % NLOCKS));
    } /* foreach fiber */
  } /* foreach slice */
}



/**
* @brief Perform TTM on the leaf mode of a 3D CSF tensor.
*
* @param csf The input tensor.
* @param tile_id Which tile.
* @param mats Input matrices.
* @param tenout Output tensor.
* @param thds Thread structures.
*/
static void p_csf_ttmc_leaf3(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{
  assert(csf->nmodes == 3);

  matrix_t const * const A = mats[csf->dim_perm[0]];
  matrix_t const * const B = mats[csf->dim_perm[1]];

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

  /* tiled outer products */
  val_t * const accum_oprod = thds[tid].scratch[2];

  /* foreach slice */
  idx_t const nslices = csf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict av = avals + (fid * rankA);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {

      /* fill fiber with outer product */
      val_t const * const restrict bv = bvals + (fids[f] * rankB);
      p_twovec_outer_prod(av, rankA, bv, rankB, accum_oprod);

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t * const restrict outv = tenout + (inds[jj] * rankA * rankB);
        omp_set_lock(locks + (inds[jj] % NLOCKS));
        for(idx_t r=0; r < rankA * rankB; ++r) {
          outv[r] += v * accum_oprod[r];
        }
        omp_unset_lock(locks + (inds[jj] % NLOCKS));
      }

    } /* foreach fiber */
  }

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



static void p_intl_decide(
    splatt_csf const * const csf,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{

  idx_t const nmodes = csf->nmodes;

  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);
    /* tile id */
    idx_t tid = 0;
    switch(csf->which_tile) {
    case SPLATT_NOTILE:
      p_csf_ttmc_intl3(csf, 0, mats, tenout, thds);
      break;

    /* XXX */
    default:
      fprintf(stderr, "SPLATT: TTM does not support tiling yet.\n");
      exit(1);
      break;
    }
    timer_stop(&thds[omp_get_thread_num()].ttime);
  }
}



static void p_leaf_decide(
    splatt_csf const * const csf,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = csf->nmodes;
  idx_t const depth = nmodes - 1;

  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);

    /* tile id */
    idx_t tid = 0;
    switch(csf->which_tile) {
    case SPLATT_NOTILE:
      p_csf_ttmc_leaf3(csf, 0, mats, tenout, thds);
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
	p_init_locks();

  idx_t nmodes = tensors[0].nmodes;
  /* find out which level in the tree this is */
  idx_t outdepth = MAX_NMODES;

  /* choose which TTM function to use */
  splatt_csf_type which = opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {

  case SPLATT_CSF_ONEMODE:
    outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
    if(outdepth == 0) {
      p_root_decide(tensors+0, mats, tenout, mode, thds, opts);
    } else if(outdepth == nmodes - 1) {
      p_leaf_decide(tensors+0, mats, tenout, mode, thds, opts);
    } else {
      p_intl_decide(tensors+0, mats, tenout, mode, thds, opts);
    }
    break;


  case SPLATT_CSF_TWOMODE:
    /* longest mode handled via second tensor's root */
    if(mode == tensors[0].dim_perm[nmodes-1]) {
      p_root_decide(tensors+1, mats, tenout, mode, thds, opts);
    /* root and internal modes are handled via first tensor */
    } else {
      outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
      if(outdepth == 0) {
        p_root_decide(tensors+0, mats, tenout, mode, thds, opts);
      } else {
        p_intl_decide(tensors+0, mats, tenout, mode, thds, opts);
      }
    }
    break;

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
  idx_t const nmodes = tt->nmodes;

  /* clear out stale results */
  p_clear_tenout(tenout, mats, nmodes, mode, tt->dims);

  val_t const * const restrict vals = tt->vals;
  val_t * mvals[MAX_NMODES];
  idx_t nfactors[MAX_NMODES];

  idx_t total_cols = 1;

  /* number of columns accumulated by the m-th mode (counting backwards) */
  idx_t ncols[MAX_NMODES+1];
  ncols[nmodes] = 1;
  val_t * buffers[MAX_NMODES];

  for(idx_t m=nmodes; m-- != 0; ) {
    nfactors[m] = mats[m]->J;
    mvals[m] = mats[m]->vals;

    /* allocate buffer */
    if(m != mode) {
      total_cols *= nfactors[m];
      ncols[m] = ncols[m+1] * nfactors[m];
      buffers[m] = splatt_malloc(ncols[m] * sizeof(**buffers));
    } else {
      ncols[m] = ncols[m+1];
      buffers[m] = NULL;
    }
  }
  assert(total_cols == ncols[0]);

  /* the last mode we accumulate with */
  idx_t const first_mode = (mode == 0) ? 1 : 0;
  idx_t const last_mode  = (mode == nmodes-1) ? nmodes-2 : nmodes-1;

  for(idx_t n=0; n < tt->nnz; ++n) {
    val_t * const restrict outrow = tenout + (tt->ind[mode][n] * total_cols);

    /* initialize buffer with nonzero value */
    val_t * curr_buff = buffers[last_mode];
    idx_t buff_size = nfactors[last_mode];
    val_t const * const restrict last_row =
        mvals[last_mode] + (tt->ind[last_mode][n] * nfactors[last_mode]);
    for(idx_t f=0; f < nfactors[last_mode]; ++f) {
      curr_buff[f] = vals[n] * last_row[f];
    }

    /* now do nmodes-1 kronecker products */
    for(idx_t m=last_mode; m-- != 0; ) {
      if(m == mode) {
        continue;
      }

      if(m != first_mode) {
        /* outer product */
        p_twovec_outer_prod(
            mvals[m] + (tt->ind[m][n] * nfactors[m]), nfactors[m],
            curr_buff, buff_size,
            buffers[m]);
      } else {
        /* accumulate into output on the first mode (last to be processed) */
        p_twovec_outer_prod_accum(
            mvals[m] + (tt->ind[m][n] * nfactors[m]), nfactors[m],
            curr_buff, buff_size,
            outrow);
      }

      /* next buffer */
      curr_buff = buffers[m];
      buff_size *= nfactors[m];
    }
  }

  for(idx_t m=0; m < nmodes; ++m) {
    splatt_free(buffers[m]);
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


void ttmc_fill_flop_tbl(
    sptensor_t * const tt,
    idx_t const * const nfactors,
    idx_t table[MAX_NMODES][MAX_NMODES])
{
  /* just assume no tiling... */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

  /* total size of output */
  idx_t ncols[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    ncols[m] = 1;
    for(idx_t d=0; d < tt->nmodes; ++d) {
      if(d != m) {
        ncols[m] *= nfactors[d];
      }
    }
  }

  /* flops if we used just CSF-1 or CSF-A */
  idx_t csf1[MAX_NMODES];
  idx_t csf2[MAX_NMODES];
  idx_t csfa[MAX_NMODES];

  idx_t const smallest_mode = argmin_elem(tt->dims, tt->nmodes);
  idx_t const largest_mode = argmax_elem(tt->dims, tt->nmodes);

  /* foreach CSF rep */
  for(idx_t i=0; i < tt->nmodes; ++i) {
    printf("MODE-%lu:  ", i);

    splatt_csf csf;
    csf_alloc_mode(tt, CSF_SORTED_MINUSONE, i, &csf, opts);

    /* foreach mode of computation */
    for(idx_t j=0; j < tt->nmodes; ++j) {

      idx_t const depth = csf_mode_depth(j, csf.dim_perm, tt->nmodes);

#define PRINT_TRAVERSAL 0
#if PRINT_TRAVERSAL
      printf("\n\ndepth: %lu\n", depth);
#endif



      idx_t flops = 0;


      /* foreach tile */
      for(idx_t tile=0; tile < csf.ntiles; ++tile) {
        idx_t out_size = nfactors[csf.dim_perm[0]];

#if PRINT_TRAVERSAL
        printf("fibs: [%lu %lu %lu]\n",
            csf.pt[tile].nfibs[0], csf.pt[tile].nfibs[1], csf.pt[tile].nfibs[2]);
        printf("perm: [%lu %lu %lu]\n",
            csf.dim_perm[0], csf.dim_perm[1], csf.dim_perm[2]);
#endif

        /* move down tree */
        for(idx_t d=1; d < depth; ++d) {
          out_size *= nfactors[csf.dim_perm[d]];
          flops += csf.pt[tile].nfibs[d] * out_size;
#if PRINT_TRAVERSAL
          printf("down (%lu): %lu x %lu [%0.3e tot]\n", d, csf.pt[tile].nfibs[d], out_size, (double)flops);
#endif
        }

        out_size = 1;

        /* move up tree */
        /* nmodes -> depth (exclusive) */
        for(idx_t d=tt->nmodes; d-- != depth+1; ) {
          out_size *= nfactors[csf.dim_perm[d]];
          flops += csf.pt[tile].nfibs[d] * out_size;
#if PRINT_TRAVERSAL
          printf("up   (%lu): %lu x %lu [%0.3e tot]\n", d, csf.pt[tile].nfibs[d], out_size, (double)flops);
#endif
        }

        /* final join if internal/leaf mode */
        if(depth > 0) {
          flops += csf.pt[tile].nfibs[depth] * ncols[j];
#if PRINT_TRAVERSAL
          printf("comb (%lu): %lu x %lu [%0.3e tot]\n", j, csf.pt[tile].nfibs[depth], ncols[j], (double) flops);
#endif
        }
      } /* end foreach tile */


      /* store result */
      table[i][j] = flops;
      printf("%0.3e  ", (double)flops);

      if(i == smallest_mode) {
        csf1[j] = flops;
        if(j != largest_mode) {
          csf2[j] = flops;
        }
      }
      if(i == j) {
        csfa[i] = flops;

        /* csf-2 uses special leaf mode */
        if(i == largest_mode) {
          csf2[j] = flops;
        }
      }
    } /* end foreach mode of computation */

    idx_t total = 0;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      total += table[i][m];
    }
    printf(" = %0.3e\n", (double)total);

    csf_free_mode(&csf);
  }
  splatt_free_opts(opts);
  printf("\n");



  /* print stats for each allocation scheme */

  idx_t total;

  /* csf-1 and csf-a */
  total = 0;
  printf("CSF-1:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("%0.3e  ", (double)csf1[m]);
    total += csf1[m];
  }
  printf(" = %0.3e\n", (double)total);
  total = 0;
  printf("CSF-2:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("%0.3e  ", (double)csf2[m]);
    total += csf2[m];
  }
  printf(" = %0.3e\n", (double)total);

  total = 0;
  printf("CSF-A:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("%0.3e  ", (double)csfa[m]);
    total += csfa[m];
  }
  printf(" = %0.3e\n", (double)total);

  bool mode_used[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mode_used[m] = false;
  }

  /* handpick best modes */
  printf("CUSTM:  ");
  total = 0;
  /* foreach mode */
  for(idx_t j=0; j < tt->nmodes; ++j) {
    idx_t best = 0;
    /* foreach csf */
    for(idx_t i=0; i < tt->nmodes; ++i) {
      if(table[i][j] <= table[best][j]) {
        best = i;
      }
    }

    mode_used[best] = true;

    total += table[best][j];
    printf("%0.3e  ", (double) table[best][j]);
  }
  printf(" = %0.3e\n", (double) total);



  /* coordinate form */
  total = 0;
  printf("COORD:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
     /* first compute nested kronecker products cost */
    idx_t nnzflops = 0;
    idx_t accum = 1;
    for(idx_t d=tt->nmodes; d-- != 0; ) {
      if(d != m) {
        accum *= nfactors[d];
        /* cost of kron at depth d */
        nnzflops += accum;
      }
    }
    printf("%0.3e  ", (double)tt->nnz * nnzflops);
    total += tt->nnz * nnzflops;
  }
  printf(" = %0.3e\n", (double)total);
  printf("\n");

  /* print CSF needed */
  printf("CUSTOM MODES:");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(mode_used[m]) {
      printf(" %lu", m);
    }
  }
  printf("\n");

}


