#ifndef SPLATT_MATRIX_H
#define SPLATT_MATRIX_H

#include "base.h"

typedef struct
{
  idx_t I;
  idx_t J;
  val_t *vals;
} matrix_t;

matrix_t * mat_rand(
  idx_t const nrows,
  idx_t const ncols);

matrix_t * mat_alloc(
  idx_t const nrows,
  idx_t const ncols);

void mat_free(
  matrix_t * mat);

#endif
