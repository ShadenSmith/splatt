#ifndef SPLATT_CSF_H
#define SPLATT_CSF_H

#include "base.h"

/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

typedef struct
{
  idx_t nmodes;
  idx_t nnz;
  idx_t dims[MAX_NMODES];
  idx_t dim_perm[MAX_NMODES];

  idx_t nfibs[MAX_NMODES];
  idx_t ** fptr;
  idx_t ** fids;
  val_t * vals;

  splatt_tile_t which_tile;
  idx_t tile_dims[MAX_NMODES];
  idx_t * tile_ptr;
} csf_t;

typedef struct
{
  csf_t * fts;

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
#define csf_alloc splatt_csf_alloc
void csf_alloc(
  csf_t * const ft,
  sptensor_t * const tt,
  double const * const splatt_opts);

#define csf_free splatt_csf_free
void csf_free(
  csf_t * const ft);

#define csf_storage splatt_csf_storage
idx_t csf_storage(
  csf_t const * const ft);

#define csf_find_mode_order splatt_csf_find_mode_order
void csf_find_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  csf_mode_type which,
  idx_t * const perm_dims);

#endif
