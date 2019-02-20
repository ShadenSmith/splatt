#ifndef SPLATT_COO_H
#define SPLATT_COO_H


#include "base.h"



/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "matrix.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


#define tt_read splatt_tt_read
/**
* @brief Load a sparse tensor from the file 'ifname'.
*
* @param ifname The file to read.
*
* @return A sparse tensor.
*/
splatt_coo * tt_read(
  char const * const ifname);


#define tt_alloc splatt_tt_alloc
/**
* @brief Allocate a sparse tensor.
*
* @param nnz The number of nonzeros to allocate.
* @param nmodes The number of modes to to allocate for.
*
* @return A pointer to the allocated tensor.
*/
splatt_coo * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes);



#define tt_fill splatt_tt_fill
/**
* @brief Store inds/vals info into tt and determine dimension information.
*        NOTE: This only stores the pointers directly, does not allocate
*        memory!
*
* @param tt The tensor to fill.
* @param nnz The length of inds/vals.
* @param nmodes The number of modes in the tensor.
* @param inds A 2D array of indices. The x'th nnz is found at ind[0][x], ...
* @param vals An array of values.
*/
void tt_fill(
  splatt_coo * const tt,
  idx_t const nnz,
  idx_t const nmodes,
  idx_t ** const inds,
  val_t * const vals);



#define tt_get_slices splatt_tt_get_slices
/**
* @brief Return a list of unique slice ids found in mode m. Slice i will be
*        included if there is a nonzero in tt the mth index equal to i.
*
* @param tt The tensor to analyze.
* @param mode The mode to operate on.
* @param nunique The number of unique slices found.
*
* @return An array at least of size nunique containing the ids of each slice
*         found in tt.
*/
idx_t * tt_get_slices(
  splatt_coo const * const tt,
  idx_t const mode,
  idx_t * nunique);


#define tt_get_hist splatt_tt_get_hist
/**
* @brief Return a histogram counting nonzeros appearing in indices of a given
*        mode.
*
* @param tt The sparse tensor to make a histogram from.
* @param mode Which mode we are counting.
*
* @return An array of length tt->dims[m].
*/
idx_t * tt_get_hist(
  splatt_coo const * const tt,
  idx_t const mode);


#define tt_free splatt_tt_free
/**
* @brief Free the fields AND pointer of a tensor.
*
* @param tt The tensor to free. NOTE: the pointer will also be freed!
*/
void tt_free(
  splatt_coo * tt);


/**
* @brief Compute the density of a sparse tensor, defined by nnz/(I*J*K).
*
* @param tt The sparse tensor.
*
* @return The density of tt.
*/
double tt_density(
  splatt_coo const * const tt);

#define tt_remove_dups splatt_tt_remove_dups
/**
* @brief Remove the duplicate entries of a tensor. Duplicate values are
*        repeatedly averaged.
*
* @param tt The modified tensor to work on. NOTE: data structures are not
*           resized!
*
* @return The number of nonzeros removed.
*/
idx_t tt_remove_dups(
  splatt_coo * const tt);


#define tt_remove_empty splatt_tt_remove_empty
/**
* @brief Relabel tensor indices to remove empty slices. Local -> global mapping
*        is written to tt->indmap.
*
* @param tt The tensor to relabel.
*
* @return The number of empty slices removed.
*/
idx_t tt_remove_empty(
  splatt_coo * const tt);


#define tt_unfold splatt_tt_unfold
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
  splatt_coo * const tt,
  idx_t const mode);


#define coo_normsq splatt_coo_normsq
/**
* @brief Compute the squared Frobenius norm of a tensor. This is the
*        sum-of-squares of all nonzeros.
*
* @param tensor The tensor to operate on.
*
* @return The norm.
*/
val_t coo_frobsq(
    splatt_coo const * const tensor);

#endif
