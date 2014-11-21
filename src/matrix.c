

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "util.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
matrix_t * mat_alloc(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = (matrix_t*) malloc(sizeof(matrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->vals = (val_t*) malloc(nrows * ncols * sizeof(val_t));
  return mat;
}

matrix_t * mat_rand(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = mat_alloc(nrows, ncols);
  val_t * const vals = mat->vals;

  for(idx_t i=0; i < nrows; ++i) {
    for(idx_t j=0; j < ncols; ++j) {
      vals[j + (i*ncols)] = rand_val();
    }
  }

  return mat;
}

void mat_free(
  matrix_t * mat)
{
  free(mat->vals);
  free(mat);
}

spmatrix_t * spmat_alloc(
  idx_t const nrows,
  idx_t const ncols,
  idx_t const nnz)
{
  spmatrix_t * mat = (spmatrix_t*) malloc(sizeof(spmatrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->nnz = nnz;
  mat->rowptr = (idx_t*) malloc((nrows+1) * sizeof(idx_t));
  mat->colind = (idx_t*) malloc(nnz * sizeof(idx_t));
  mat->vals   = (val_t*) malloc(nnz * sizeof(val_t));
  return mat;
}

void spmat_free(
  spmatrix_t * mat)
{
  mat->I = 0;
  mat->J = 0;
  mat->nnz = 0;
  free(mat->rowptr);
  free(mat->colind);
  free(mat->vals);
  free(mat);
}

