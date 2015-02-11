#ifndef SPLATT_SPTENSOR_H
#define SPLATT_SPTENSOR_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"



/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

/**
* @brief Types of tensors supported by splatt.
*/
typedef enum
{
  SPLATT_3MODE,   /** Three-mode tensors. */
  SPLATT_NMODE,   /** Tensors of with arbitrary numbers of modes.
                      NOTE: support is minimal. */
} tt_type;


/**
* @brief The main data structure for representing sparse tensors in
*        coordinate format.
*/
typedef struct
{
  tt_type type;   /** Type of tensor represented */
  idx_t nmodes;   /** The number of modes in the tensor, denoted 'm'. */
  idx_t nnz;      /** The number of nonzeros in the tensor. */
  idx_t * dims;   /** An array containing the dimension of each mode. */
  idx_t ** ind;   /** An m x nnz matrix containing the coordinates of each
                      nonzero. The nth nnz is accessed via ind[0][n], ind[1][n],
                      ..., ind[m][n]. */
  val_t * vals;   /** An array containing the values of each nonzero. */
  int tiled;      /** Whether sptensor_t has been tiled. Used by ftensor_t. */

  idx_t * indmap[MAX_NMODES]; /** Maps local -> global indices. */
} sptensor_t;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


/**
* @brief Load a sparse tensor from the file 'ifname'.
*
* @param ifname The file to read.
*
* @return A sparse tensor.
*/
sptensor_t * tt_read(
  char const * const ifname);


/**
* @brief Allocate a sparse tensor.
*
* @param nnz The number of nonzeros to allocate.
* @param nmodes The number of modes to to allocate for.
*
* @return A pointer to the allocated tensor.
*/
sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes);


/**
* @brief Free the fields AND pointer of a tensor.
*
* @param tt The tensor to free. NOTE: the pointer will also be freed!
*/
void tt_free(
  sptensor_t * tt);


/**
* @brief Unfold a tensor to a sparse matrix in CSR format.
*
* @param tt The tensor to unfold.
* @param mode The mode unfolding to operate on.
*
* @return The unfolded tensor in CSR format. The matrix will be of dimension
*         dims[mode] x (dims[0] * dims[1] * ... * dims[mode-1] *
*         dims[mode+1] * ... * dims[m].
*/
spmatrix_t * tt_unfold(
  sptensor_t * const tt,
  idx_t const mode);

#endif
