
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
    splatt_csf_t const * const tensor,
    splatt_val_t ** matrices,
    splatt_val_t * const tenout,
    double const * const options)
{
  idx_t const nmodes = tensor->nmodes;

  idx_t maxcols = 0;

  /* fill matrix pointers  */
  matrix_t * mats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (matrix_t *) malloc(sizeof(matrix_t));
    mats[m]->I = tensor->dims[m];
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

  ttm_splatt(tensor, mats, tenout, mode, thds, nthreads);

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
static inline void __outer_prod(
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


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void ttm_splatt(
  splatt_csf_t const * const ft,
  matrix_t ** mats,
  val_t * const restrict tenout,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];
  idx_t const nslices = ft->dims[mode];

  idx_t const rankA = A->J;
  idx_t const rankB = B->J;

  memset(tenout, 0, nslices * rankA * rankB * sizeof(val_t));

  idx_t const * const restrict sptr = ft->sptr;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  val_t const * const restrict vals = ft->vals;

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    timer_start(&thds[tid].ttime);

    /* foreach slice */
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      val_t * const restrict outv = tenout + (s * rankA * rankB);

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
        //__outer_prod(av, rankA, accumF, rankB, outv);
        __outer_prod(accumF, rankB, av, rankA, outv);
      }
    }
    timer_stop(&thds[tid].ttime);
  } /* end omp parallel */
}

