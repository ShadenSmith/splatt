

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "spmatrix.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
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

