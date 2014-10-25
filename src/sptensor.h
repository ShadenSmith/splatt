#ifndef SPLATT_SPTENSOR_H
#define SPLATT_SPTENSOR_H

#include "base.h"

typedef struct
{
  idx_t dims[NMODES];
  idx_t * ind[NMODES];
  idx_t nnz;
  val_t * vals;
} sptensor_t;

#endif
