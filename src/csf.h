#ifndef SPLATT_CSF_H
#define SPLATT_CSF_H

#include "base.h"

/******************************************************************************
 * ENUMS and CONSTANTS
 *****************************************************************************/


/* The types of mode ordering available. */
typedef enum
{
  CSF_SORTED_SMALLFIRST, /** sort the modes in non-decreasing order */
  CSF_SORTED_BIGFIRST,   /** sort the modes in non-increasing order */
  CSF_INORDER_MINUSONE,  /** one mode is placed first, rest naturally ordered*/
  CSF_SORTED_SMALLFIRST_MINUSONE, /** one placed first, rest sorted by size */
  CSF_SORTED_BIGFIRST_MINUSONE, /** one placed first, rest sorted by size */
  CSF_SORTED_MINUSONE,   /** one mode is placed first, rest sorted by size */
  CSF_CUSTOM             /** custom mode ordering. dim_perm must be set! */
} csf_mode_type;


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "sptensor.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define csf_alloc splatt_csf_alloc
/**
* @brief Convert a coordinate tensor to CSF form. Options will determine how
*        many tensors to allocate and which tiling scheme to use.
*
* @param tt The coordinate tensor to convert from.
* @param opts 'SPLATT_OPTION_CSF_ALLOC' and 'SPLATT_OPTION_TILE' determine
*             the allocation scheme.
*
* @return The allocated tensor(s).
*/
splatt_csf * csf_alloc(
  sptensor_t * const tt,
  double const * const opts);


#define csf_alloc_mode splatt_csf_alloc_mode
/**
* @brief Convert a coordinate tensor to CSF form, optimized for a certain
*        mode. The tensor is written to 'csf', whose pointer is assumed to be
*        non-NULL. Exactly one tensor will be filled, unlike csf_alloc().
*
* @param tt The coordinate tensor to convert from.
* @param which_ordering Which ordering scheme to use.
* @param mode_special Which mode to make the root mode.
* @param csf The tensor to fill.
* @param opts 'SPLATT_OPTION_TILE' determines the allocation scheme.
*/
void csf_alloc_mode(
  sptensor_t * const tt,
  csf_mode_type which_ordering,
  idx_t const mode_special,
  splatt_csf * const csf,
  double const * const opts);


#define csf_free splatt_csf_free
/**
* @brief Free all memory allocated for a tensor in CSF form. This should be
*        paired with csf_alloc().
*
* @param csf The tensor to free.
* @param opts opts[SPLATT_OPTION_CSF_ALLOC] tells us how many tensors are
*             allocated.
*/
void csf_free(
  splatt_csf * const csf,
  double const * const opts);


/**
* @brief Free the memory allocated for one CSF representation. This should be
*        paired with csf_alloc_mode().
*
* @param csf The tensor to free.
*/
void csf_free_mode(
    splatt_csf * const csf);



/**
* @brief Return the number of tensors allocated.
*
* @param tensors The allocated tensor(s).
* @param opts The options during allocation.
*
* @return The number of tensors.
*/
idx_t csf_ntensors(
    splatt_csf const * const tensors,
    double const * const opts);


#define csf_storage splatt_csf_storage
/**
* @brief Compute the number of bytes requiredto store a tensor.
*
* @param tensors The tensor(s) to compute storage information of.
* @param opts opts[SPLATT_OPTION_CSF_ALLOC] tells us how many tensors are
*             allocated.
*
* @return The size, in bytes, of the tensor(s) in CSF form.
*/
size_t csf_storage(
  splatt_csf const * const tensors,
  double const * const opts);


#define csf_frobsq splatt_csf_frobsq
/**
* @brief Compute the squared Frobenius norm of a tensor. This is the
*        sum-of-squares of all nonzeros.
*
* @param tensor The tensor to operate on.
*
* @return The norm.
*/
val_t csf_frobsq(
  splatt_csf const * const tensor);


#define csf_find_mode_order splatt_csf_find_mode_order
/**
* @brief Find an ordering of tensor modes based on dimensions.
*
* @param dims The dimensions of the tensor.
* @param nmodes The number of modes.
* @param which Which ordering to use.
* @param mode Which mode to focus on (if applicable e.g., CSF_SORTED_MINUSONE).
* @param[out] perm_dims is filled with the mode permutation. perm_dims[m]
*                  specifies which mode will be the m-th mode operated on.
*/
void csf_find_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  csf_mode_type which,
  idx_t const mode,
  idx_t * const perm_dims);


#define csf_mode_depth splatt_csf_mode_depth
/**
* @brief Map a mode (in the input system) to the tree level that it is found.
*        This is equivalent to the inverse dim_perm.
*
* @param mode The mode (relative to the input) to lookup.
* @param perm The dimenison permutation.
* @param nmodes The number of modes.
*
* @return The level of the tree that mode is mapped to.
*/
static inline idx_t csf_mode_depth(
  idx_t const mode,
  idx_t const * const perm,
  idx_t const nmodes)
{
  for(idx_t m=0; m < nmodes; ++m) {
    if(perm[m] == mode) {
      return m;
    }
  }

  /* XXX: ERROR */
  assert(1 == 2);
  return MAX_NMODES;
}

#endif
