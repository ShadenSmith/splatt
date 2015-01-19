
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"
#include "thd_info.h"
#include "tile.h"
#include <omp.h>

#define dmin(x,y) ((x) < (y) ? (x) : (y))


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


/******************************************************************************
 * SPLATT MTTKRP
 *****************************************************************************/
void mttkrp_splatt(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  if(ft->tiled) {
    mttkrp_splatt_coop_tiled(ft, mats, mode, thds, nthreads);
    return;
  }

  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perms[mode][1]];
  matrix_t const * const B = mats[ft->dim_perms[mode][2]];
  idx_t const nslices = ft->dims[mode];
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict sptr = ft->sptr[mode];
  idx_t const * const restrict fptr = ft->fptr[mode];
  idx_t const * const restrict fids = ft->fids[mode];
  idx_t const * const restrict inds = ft->inds[mode];
  val_t const * const restrict vals = ft->vals[mode];

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch;
    timer_start(&thds[tid].ttime);

    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      val_t * const restrict mv = mvals + (s * rank);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t const * const restrict av = avals  + (fids[f] * rank);
        for(idx_t r=0; r < rank; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }
    }

    timer_stop(&thds[tid].ttime);
  } /* end parallel region */
}


void mttkrp_splatt_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perms[mode][1]];
  matrix_t const * const B = mats[ft->dim_perms[mode][2]];

  idx_t const nslabs = ft->nslabs[mode];
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict slabptr = ft->slabptr[mode];
  idx_t const * const restrict sids = ft->sids[mode];
  idx_t const * const restrict fptr = ft->fptr[mode];
  idx_t const * const restrict fids = ft->fids[mode];
  idx_t const * const restrict inds = ft->inds[mode];
  val_t const * const restrict vals = ft->vals[mode];

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch;
    timer_start(&thds[tid].ttime);

    #pragma omp for schedule(dynamic, 1) nowait
    for(idx_t s=0; s < nslabs; ++s) {
      /* foreach fiber in slice */
      for(idx_t f=slabptr[s]; f < slabptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t       * const restrict mv = mvals + (sids[f] * rank);
        val_t const * const restrict av = avals + (fids[f] * rank);
        for(idx_t r=0; r < rank; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }
    }

    timer_stop(&thds[tid].ttime);
  } /* end parallel region */
}


void mttkrp_splatt_coop_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perms[mode][1]];
  matrix_t const * const B = mats[ft->dim_perms[mode][2]];

  idx_t const nslabs = ft->nslabs[mode];
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict slabptr = ft->slabptr[mode];
  idx_t const * const restrict sptr = ft->sptr[mode];
  idx_t const * const restrict sids = ft->sids[mode];
  idx_t const * const restrict fptr = ft->fptr[mode];
  idx_t const * const restrict fids = ft->fids[mode];
  idx_t const * const restrict inds = ft->inds[mode];
  val_t const * const restrict vals = ft->vals[mode];

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch;
    val_t * const restrict localm = (val_t *) thds[tid].scratch2;
    timer_start(&thds[tid].ttime);

    /* foreach slab */
    for(idx_t s=0; s < nslabs; ++s) {
      /* foreach fiber in slab */
      #pragma omp for schedule(dynamic, 8)
      for(idx_t sl=slabptr[s]; sl < slabptr[s+1]; ++sl) {
        idx_t const slice = sids[sl];
        for(idx_t f=sptr[sl]; f < sptr[sl+1]; ++f) {
          /* first entry of the fiber is used to initialize accumF */
          idx_t const jjfirst  = fptr[f];
          val_t const vfirst   = vals[jjfirst];
          val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] = vfirst * bv[r];
          }

          /* foreach nnz in fiber */
          for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
            val_t const v = vals[jj];
            val_t const * const restrict bv = bvals + (inds[jj] * rank);
            for(idx_t r=0; r < rank; ++r) {
              accumF[r] += v * bv[r];
            }
          }

          /* scale inner products by row of A and update thread-local M */
          val_t       * const restrict mv = localm + ((slice % TILE_SIZES[0]) * rank);
          val_t const * const restrict av = avals + (fids[f] * rank);
          for(idx_t r=0; r < rank; ++r) {
            mv[r] += accumF[r] * av[r];
          }
        }
      }

#if 0
      /* reduction on localm into M */
      #pragma omp master
      {
        for(idx_t t=0; t < nthreads; ++t) {
          val_t * const restrict localm = (val_t *) thds[t].scratch2;
          for(idx_t i=0; i < TILE_SIZES[0]; ++i) {
            /* map i back to global slice id */
            idx_t const globalrow = i + (s * TILE_SIZES[0]);
            if(globalrow >= ft->dims[mode]) {
              break;
            }
            for(idx_t r=0; r < rank; ++r) {
              mvals[r + (globalrow*rank)] += localm[r + (i*rank)];
              localm[r + (i*rank)] = 0;
            }
          }
        }
      } /* end reduction */
      #pragma omp barrier
#else
      idx_t const start = s * TILE_SIZES[0];
      idx_t const stop  = dmin((s+1) * TILE_SIZES[0], ft->dims[mode]);

      #pragma omp for schedule(static)
      for(idx_t i=start; i < stop; ++i) {
        /* map i back to global slice id */
        idx_t const localrow = i % TILE_SIZES[0];
        for(idx_t t=0; t < nthreads; ++t) {
          val_t * const restrict localm = (val_t *) thds[t].scratch2;
          for(idx_t r=0; r < rank; ++r) {
            mvals[r + (i*rank)] += localm[r + (localrow*rank)];
            localm[r + (localrow*rank)] = 0.;
          }
        }
      }
#endif

    } /* end foreach slab */
    timer_stop(&thds[tid].ttime);
  } /* end omp parallel */
}



/******************************************************************************
 * GIGA MTTKRP
 *****************************************************************************/
void mttkrp_giga(
  spmatrix_t const * const spmat,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  idx_t const I = spmat->I;
  idx_t const rank = M->J;

  idx_t const * const restrict rowptr = spmat->rowptr;
  idx_t const * const restrict colind = spmat->colind;
  val_t const * const restrict vals   = spmat->vals;

  for(idx_t r=0; r < rank; ++r) {
    val_t       * const restrict mv =  M->vals + (r * I);
    val_t const * const restrict av =  A->vals + (r * A->I);
    val_t const * const restrict bv =  B->vals + (r * B->I);

    /* Joined Hadamard products of X, C, and B */
    #pragma omp parallel for schedule(dynamic, 16)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
        idx_t const a = colind[y] / B->I;
        idx_t const b = colind[y] % B->I;
        scratch[y] = vals[y] * av[a] * bv[b];
      }
    }

    /* now accumulate rows into column of M1 */
    #pragma omp parallel for schedule(dynamic, 16)
    for(idx_t i=0; i < I; ++i) {
      val_t sum = 0;
      for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
        sum += scratch[y];
      }
      mv[i] = sum;
    }
  }
}


/******************************************************************************
 * TTBOX MTTKRP
 *****************************************************************************/
void mttkrp_ttbox(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  idx_t const I = tt->dims[mode];
  idx_t const rank = M->J;

  memset(M->vals, 0, I * rank * sizeof(val_t));

  val_t * const restrict m1vals = M->vals;

  idx_t const nnz = tt->nnz;
  idx_t const * const restrict indM = tt->ind[mode];
  idx_t const * const restrict indA =
    mode == 0 ? tt->ind[1] : tt->ind[0];
  idx_t const * const restrict indB =
    mode == 2 ? tt->ind[1] : tt->ind[2];

  val_t const * const restrict vals = tt->vals;

  for(idx_t r=0; r < rank; ++r) {
    val_t       * const restrict mv =  M->vals + (r * I);
    val_t const * const restrict av =  A->vals + (r * A->I);
    val_t const * const restrict bv =  B->vals + (r * B->I);

    /* stretch out columns of A and B */
    #pragma omp parallel for
    for(idx_t x=0; x < nnz; ++x) {
      scratch[x] = vals[x] * av[indA[x]] * bv[indB[x]];
      //scratch[x] = vals[x] * A->vals[r + (rank*indA[x])] * B->vals[r + (rank*indB[x])];
    }

    /* now accumulate into m1 */
    for(idx_t x=0; x < nnz; ++x) {
      mv[indM[x]] += scratch[x];
      //M->vals[r + (rank * indM[x])] += scratch[x];
    }
  }
}



