#ifndef SPLATT_SVD_H
#define SPLATT_SVD_H

#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

typedef struct
{
  /* matrices */
  val_t * A;
  val_t * S;
  val_t * U;
  val_t * Vt;

  /* workspace */
  int lwork;
  val_t * workspace;
  int * iwork;
} svd_ws;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

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
    val_t const * const inmat,
    val_t * outmat,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const rank,
    svd_ws * const ws);


#define alloc_svd_ws splatt_alloc_svd_ws
/**
* @brief Allocate and initialize the required workspace for SVDs.
*
* @param ws The workspace to initialize.
* @param nmats The number of SVDs that will be performed per itation.
* @param nrows The rows in the SVDs.
* @param ncolumns The columns in the SVDs.
*/
void alloc_svd_ws(
    svd_ws * const ws,
    idx_t const nmats,
    idx_t const * const nrows,
    idx_t const * const ncolumns);


#define free_svd_ws splatt_free_svd_ws
/**
* @brief Free all memory allocate for an SVD workspace.
*
* @param ws The workspace to free.
*/
void free_svd_ws(
    svd_ws * const ws);


#endif
