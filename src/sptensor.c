
#include "sptensor.h"

sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes)
{
  sptensor_t * tt = (sptensor_t*) malloc(sizeof(sptensor_t));

  tt->nnz = nnz;
  tt->vals = (val_t*) malloc(nnz * sizeof(val_t));

  tt->nmodes = nmodes;
  tt->type = (nmodes == 3) ? SPLATT_3MODE : SPLATT_NMODE;

  tt->dims = (idx_t*) malloc(nmodes * sizeof(idx_t));
  tt->ind = (idx_t**) malloc(nmodes * sizeof(idx_t*));
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
  free(tt->dims);
  free(tt->ind);
  free(tt->vals);
}

static void __ttqsort(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const n)
{

}

void tt_sort(
  sptensor_t * const tt,
  idx_t const mode)
{
  idx_t * cmplt = (idx_t*) malloc(tt->nmodes * sizeof(idx_t));
  cmplt[0] = mode;


  free(cmplt);
}

