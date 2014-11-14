#ifndef SPLATT_SPTENSOR_H
#define SPLATT_SPTENSOR_H

#include "base.h"

typedef struct
{
  idx_t nnz;
  idx_t dims[NMODES];
  idx_t * ind[NMODES];
  val_t * vals;
} sptensor_t;

sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const dims[NMODES]);

void tt_free(
  sptensor_t * tt);

#endif
