#ifndef SPLATT_SVD_H
#define SPLATT_SVD_H

#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

typedef struct
{
  val_t * A;
  val_t * S;
  val_t * U;
  val_t * Vt;

  int lwork;
  val_t * workspace;
  int * iwork;
} svd_ws;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define optimal_svd_work_size splatt_optimal_svd_work_size
/**
* @brief Compute the optimal workspace size for LAPACK_SVD.
*
* @param nrows The number of rows.
* @param ncols The number of columns.
*
* @return The optimal workspace size.
*/
int optimal_svd_work_size(
    idx_t const nrows,
    idx_t const ncols);


void alloc_svd_ws(
    val_t ** svdbuf,
    val_t ** U,
    val_t ** S,
    val_t ** Vt,
    val_t ** lwork,
    int ** iwork,
    idx_t const * const nrows,
    idx_t const * const ncolumns);


#define left_singulars splatt_left_singulars
/**
* @brief Compute the SVD of a row-major matrix and return the first 'rank' of
*        the left singular vectors.
*
* @param inmat The matrix to factor.
* @param[out] outmat A row-major matrix of the first 'rank' left singular vecs.
* @param nrows The number of rows of 'inmat'.
* @param ncols The number of columns of 'inmat'.
* @param rank The desired number of columns.
*/
void left_singulars(
    val_t * inmat,
    val_t * outmat,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const rank);

#endif
