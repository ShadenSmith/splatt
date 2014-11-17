#ifndef SPLATT_SPTENSOR_H
#define SPLATT_SPTENSOR_H

#include "base.h"

#include "spmatrix.h"

typedef enum
{
  SPLATT_NMODE,
  SPLATT_3MODE
} tt_type;

typedef struct
{
  tt_type type;
  idx_t nnz;
  idx_t nmodes;
  idx_t * dims;
  idx_t ** ind;
  val_t * vals;
} sptensor_t;

sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes);

void tt_free(
  sptensor_t * tt);

spmatrix_t * tt_unfold(
  sptensor_t * const tt,
  idx_t const mode);

#endif
