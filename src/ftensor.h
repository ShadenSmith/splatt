#ifndef SPLATT_FTENSOR_H
#define SPLATT_FTENSOR_H

#include "base.h"
#include "matrix.h"

typedef struct
{
  idx_t dims[3];
  idx_t nnz;
  idx_t nparts;
  idx_t * partptr[3];
  idx_t * slptr[3];
  idx_t * fptr[3];
  idx_t * fid[3];
  idx_t * ind[3];
  val_t * vals[3];
  int direction[3];
} ftensor_t;

typedef struct
{
  idx_t dims[3];
  idx_t nnz;
  idx_t * indI;
  idx_t * indJ;
  idx_t * indK;
  val_t * vals;
} sptensor_t;

void ttkrao_ftensor(
  ftensor_t const * const tt,
  idx_t const mode,
  matrix_t const * const A,
  matrix_t const * const B,
  matrix_t const * const M1);

#endif
