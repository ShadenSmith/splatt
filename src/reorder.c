
#include "reorder.h"

void tt_perm(
  sptensor_t * const tt,
  idx_t const * const perm)
{
  idx_t * ind;
  idx_t const * _perm = perm;
  idx_t const nnz = tt->nnz;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ind = tt->ind[m];
    for(idx_t n=0; n < nnz; ++n) {
      ind[n] = _perm[ind[n]];
    }
    _perm += tt->dims[m];
  }
}

