#ifndef SPLATT_CSF_H
#define SPLATT_CSF_H

#include "base.h"

/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

typedef struct
{
  idx_t nfibs[MAX_NMODES];
  idx_t * fptr[MAX_NMODES];
  idx_t * fids[MAX_NMODES];
  val_t * vals;
} csf_sparsity_t;


typedef struct
{
  idx_t nnz;
  idx_t nmodes;
  idx_t dims[MAX_NMODES];
  idx_t dim_perm[MAX_NMODES];

  splatt_tile_t which_tile;
  idx_t ntiles;
  idx_t tile_dims[MAX_NMODES];

  csf_sparsity_t * pt; /** sparsity structure -- one for each tile */
} csf_t;


/* The types of mode ordering available. */
typedef enum
{
  CSF_SORTED_SMALLFIRST, /** sort the modes in non-decreasing order */
  CSF_SORTED_BIGFIRST,   /** sort the modes in non-increasing order */
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
void csf_alloc(
  csf_t * const ct,
  sptensor_t * const tt,
  double const * const opts);

#define csf_free splatt_csf_free
void csf_free(
  csf_t * const ct);

#define csf_storage splatt_csf_storage
idx_t csf_storage(
  csf_t const * const ct);

#define csf_find_mode_order splatt_csf_find_mode_order
void csf_find_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  csf_mode_type which,
  idx_t * const perm_dims);


/**
* @brief Map a mode (in the input system) to the tree level that it is found.
*        This is equivalent to a linear-time lookup in the inverse dim_perm.
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
  return MAX_NMODES;
}

#endif
