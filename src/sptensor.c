
#include "sptensor.h"

sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const dims[NMODES])
{
  sptensor_t * tt = (sptensor_t*) malloc(sizeof(sptensor_t));

  tt->nnz = nnz;
  tt->vals = (val_t*) malloc(nnz * sizeof(val_t));
  for(idx_t m=0; m < NMODES; ++m) {
    tt->dims[m] = dims[m];
    tt->ind[m] = (idx_t*) malloc(nnz * sizeof(idx_t));
  }

  return tt;
}

void tt_free(
  sptensor_t * tt)
{
  tt->nnz = 0;
  for(idx_t m=0; m < NMODES; ++m) {
    tt->dims[m] = 0;
    free(tt->ind[m]);
  }
  free(tt->vals);
}

