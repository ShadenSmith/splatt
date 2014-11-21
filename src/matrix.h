#ifndef SPLATT_MATRIX_H
#define SPLATT_MATRIX_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  idx_t I;
  idx_t J;
  val_t *vals;
} matrix_t;

typedef struct
{
  idx_t I;
  idx_t J;
  idx_t nnz;
  idx_t * rowptr;
  idx_t * colind;
  val_t * vals;
} spmatrix_t;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
matrix_t * mat_rand(
  idx_t const nrows,
  idx_t const ncols);

matrix_t * mat_alloc(
  idx_t const nrows,
  idx_t const ncols);

void mat_free(
  matrix_t * mat);

spmatrix_t * spmat_alloc(
  idx_t const nrows,
  idx_t const ncols,
  idx_t const nnz);

void spmat_free(
  spmatrix_t * mat);

#endif
