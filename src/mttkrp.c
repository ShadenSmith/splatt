
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void splatt_mttkrp(
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

