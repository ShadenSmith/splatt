#ifndef SPLATT_SPMATRIX_H
#define SPLATT_SPMATRIX_H

#include "base.h"

typedef struct
{
  idx_t I;
  idx_t J;
  idx_t nnz;
  idx_t * rowptr;
  idx_t * colind;
  val_t * vals;
} spmatrix_t;

spmatrix_t * spmat_alloc(
  idx_t const nrows,
  idx_t const ncols,
  idx_t const nnz);

void spmat_free(
  spmatrix_t * mat);

#endif
