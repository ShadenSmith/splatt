
#include "sptensor.h"

sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes)
{
  sptensor_t * tt = (sptensor_t*) malloc(sizeof(sptensor_t));

  if(nmodes == 3) {
    tt->type = SPLATT_3MODE;
  } else {
    tt->type = SPLATT_NMODE;
  }

  tt->nnz = nnz;
  tt->nmodes = nmodes;
  tt->vals = (val_t*) malloc(nnz * sizeof(val_t));

  tt->ind = (idx_t**) malloc(nmodes * sizeof(idx_t*));
  tt->dims = (idx_t*) malloc(nmodes * sizeof(idx_t));
  for(idx_t m=0; m < nmodes; ++m) {
    tt->ind[m] = (idx_t*) malloc(nnz * sizeof(idx_t));
  }

  return tt;
}

void tt_free(
  sptensor_t * tt)
{
  tt->nnz = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    free(tt->ind[m]);
  }
  tt->nmodes = 0;
  free(tt->ind);
  free(tt->vals);
}

