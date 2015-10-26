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
  CSF_SORTED_MINUSONE    /** one mode is placed first, rest sorted  */
} csf_mode_type;


/**
* @brief Only tile modes at least this depth in the tree.
*        NOTE: 0-indexed! So, depth=1 will tile all but the top level modes.
*/
static idx_t const MIN_TILE_DEPTH = 1;


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "sptensor.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
#define csf_alloc splatt_csf_alloc
splatt_csf * splatt_csf_alloc(
  sptensor_t * const tt,
  double const * const opts);

#define csf_free splatt_csf_free
void csf_free(
  splatt_csf * const ct,
  double const * const opts);

#define csf_storage splatt_csf_storage
size_t csf_storage(
  splatt_csf const * const tensors,
  double const * const opts);

#define csf_find_mode_order splatt_csf_find_mode_order
void csf_find_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  csf_mode_type which,
  idx_t const mode,
  idx_t * const perm_dims);


#define csf_frobsq splatt_csf_frobsq
val_t csf_frobsq(
  splatt_csf const * const tensor);


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
