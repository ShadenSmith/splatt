#ifndef SPLATT_MATRIX_H
#define SPLATT_MATRIX_H

#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  idx_t I;
  idx_t J;
  val_t *vals;
  int rowmajor;
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

typedef enum
{
  MAT_NORM_2,
  MAT_NORM_MAX
} splatt_mat_norm;


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "splatt_mpi.h"
#include "thd_info.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
/**
* @brief Compute the Cholesky factorization of A.
*
* @param A The SPD matrix A.
* @param L The lower-triangular result.
*/
void mat_cholesky(
  matrix_t const * const A,
  matrix_t * const L);


/**
* @brief Dense matrix-matrix multiplication, C = AB + C.
*
* @param A The left multiplication parameter.
* @param B The right multiplication parameter.
* @param C The result matrix. NOTE: C is not zeroed before multiplication!
*/
void mat_matmul(
  matrix_t const * const A,
  matrix_t const * const B,
  matrix_t  * const C);

/**
* @brief Compute the 'inverse' of symmetric matrix A.
*
* @param A Symmetric FxF matrix.
* @param Abuf FxF buffer space.
*/
void mat_syminv(
  matrix_t * const A);

/**
* @brief Compute (A^T A * B^T B * C^T C ...) where * is the Khatri-Rao product.
*
* @param mats An array of matrices.
* @param start The first matrix to include.
* @param end The last matrix to include. This can be before start because we
*            operate modulo nmats.
* @param nmats The number of matrices.
* @param ret The FxF output matrix.
*/
void mat_aTa_hada(
  matrix_t ** mats,
  idx_t const start,
  idx_t const end,
  idx_t const nmats,
  matrix_t * const buf,
  matrix_t * const ret);


/**
* @brief Compute A^T * A with a nice row-major pattern.
*
* @param A The input matrix.
* @param ret The output matrix, A^T * A.
* @param thds Data structure for thread scratch space.
*/
void mat_aTa(
  matrix_t const * const A,
  matrix_t * const ret,
  rank_info * const rinfo,
  thd_info * const thds,
  idx_t const nthreads);


/**
* @brief Normalize the columns of A and return the norms in lambda.
*        Supported norms are:
*          1. 2-norm
*          2. max-norm
*
* @param A The matrix to normalize.
* @param lambda The vector of column norms.
* @param which Which norm to use.
*/
void mat_normalize(
  matrix_t * const A,
  val_t * const restrict lambda,
  splatt_mat_norm const which,
  rank_info * const rinfo,
  thd_info * const thds,
  idx_t const nthreads);

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

matrix_t * mat_mkrow(
  matrix_t const * const mat);
matrix_t * mat_mkcol(
  matrix_t const * const mat);

#endif
