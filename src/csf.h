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
} ctensor_t;


/* The types of mode ordering available. */
typedef enum
{
  CSF_SORTED_SMALLFIRST, /** sort the modes in non-decreasing order */
  CSF_SORTED_BIGFIRST,   /** sort the modes in non-increasing order */
} csf_mode_type;


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "sptensor.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
#define ctensor_alloc splatt_ctensor_alloc
void ctensor_alloc(
  ctensor_t * const ct,
  sptensor_t * const tt,
  double const * const opts);

#define ctensor_free splatt_ctensor_free
void ctensor_free(
  ctensor_t * const ct);

#if 0
#define csf_storage splatt_csf_storage
idx_t csf_storage(
  csf_t const * const ft);
#endif

#define csf_find_mode_order splatt_csf_find_mode_order
void csf_find_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  csf_mode_type which,
  idx_t * const perm_dims);

#endif
