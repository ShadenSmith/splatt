
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mttkrp_splatt(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode)
{
  matrix_t * const m1 = mats[mode];
  idx_t const nslices = m1->I;
  idx_t const rank = m1->J;

  val_t * const m1vals = m1->vals;
  val_t const * const avals = mats[ft->dim_perms[mode][1]]->vals;
  val_t const * const bvals = mats[ft->dim_perms[mode][2]]->vals;
  memset(m1vals, 0, nslices * rank * sizeof(val_t));

  idx_t const * const restrict sptr = ft->sptr[mode];
  idx_t const * const restrict fptr = ft->fptr[mode];
  idx_t const * const restrict fids = ft->fids[mode];
  idx_t const * const restrict inds = ft->inds[mode];
  val_t const * const restrict vals = ft->vals[mode];

  val_t * const restrict accumF = (val_t *) malloc(rank * sizeof(val_t));

  for(idx_t s=0; s < nslices; ++s) {
    val_t * const restrict m1 = m1vals + (s * rank);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict av = avals + (inds[jjfirst] * rank);
      for(idx_t r=0; r < rank; ++r) {
        accumF[r] = vfirst * av[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict av = avals + (inds[jj] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] += v * av[r];
        }
      }

      val_t const * const restrict bv = bvals  + (fids[f] * rank);
      for(idx_t r=0; r < rank; ++r) {
        m1[r] += accumF[r] * bv[r];
      }
    }
  }

  free(accumF);
}


void mttkrp_ttbox(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t * const m1 = mats[mode];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  idx_t const rank = m1->J;

  val_t * const restrict m1vals = m1->vals;

  idx_t const nnz = tt->nnz;
  idx_t const * const restrict indM = tt->ind[mode];
  idx_t const * const restrict indA =
    mode == 0 ? tt->ind[1] : tt->ind[0];
  idx_t const * const restrict indB =
    mode == 2 ? tt->ind[1] : tt->ind[2];

  val_t const * const restrict vals = tt->vals;

  for(idx_t r=0; r < rank; ++r) {
    val_t       * const restrict mv = m1->vals + (r * m1->I);
    val_t const * const restrict av =  A->vals + (r * A->I);
    val_t const * const restrict bv =  B->vals + (r * B->I);

    /* stretch out columns of A and B */
    for(idx_t x=0; x < nnz; ++x) {
      scratch[x] = vals[x] * av[indA[x]] * bv[indB[x]];
    }

    /* now accumulate into m1 */
    for(idx_t x=0; x < nnz; ++x) {
      mv[indM[x]] += scratch[x];
    }
  }
}




