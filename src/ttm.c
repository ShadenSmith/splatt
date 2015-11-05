
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "ttm.h"
#include "thd_info.h"
#include "tile.h"
#include <omp.h>


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

int splatt_ttm(
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

  printf("extracted\n");

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  idx_t const nthreads = (idx_t) options[SPLATT_OPTION_NTHREADS];
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 1,
    (maxcols * sizeof(val_t)) + 64);

  printf("thread\n");

  ttm_csf(tensors, mats, tenout, mode, thds, options);

  printf("cleanup\n");

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
* @param out The output matrix which is (nA x nB).
*/
static inline void p_outer_prod(
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
static void p_csf_ttm_root(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{
  if(csf->nmodes != 3) {
    fprintf(stderr, "SPLATT: TTM only supports 3 modes right now.\n");
    exit(1);
  }

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
  val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];

  /* foreach slice */
  idx_t const nslices = csf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    val_t * const restrict outv = tenout + (fid * rankA * rankB);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * rankB);
      for(idx_t r=0; r < rankB; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * rankB);
        for(idx_t r=0; r < rankB; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      /* accumulate outer product into tenout */
      val_t const * const restrict av = avals  + (fids[f] * rankA);
      //p_outer_prod(av, rankA, accumF, rankB, outv);
      p_outer_prod(accumF, rankB, av, rankA, outv);
    }
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
      p_csf_ttm_root(tensor, 0, mats, tenout, thds);
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


void ttm_csf(
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



void ttm_stream(
    sptensor_t const * const tt,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    double const * const opts)
{
  /* clear out stale results */
  p_clear_tenout(tenout, mats, tt->nmodes, mode, tt->dims);

  omp_set_num_threads(opts[SPLATT_OPTION_NTHREADS]);


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

  /* buffer to accumulate nonzero into */
  val_t * accum = malloc(ncols * sizeof(*accum));

  for(idx_t n=0; n < tt->nnz; ++n) {
    memset(accum, 0, ncols * sizeof(*accum));

    /* write val to accum */

    for(idx_t m=0; m < tt->nmodes; ++m) {
      if(m == mode) {
        continue;
      }

      val_t const * const restrict inrow = mvals[m] +
          (tt->ind[m][n] * nfactors[m]);
    }

    val_t * const restrict outrow = tenout + (tt->ind[mode][n] * ncols);
    for(idx_t f=0; f < ncols; ++f) {
      outrow[f] += accum[f];
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


