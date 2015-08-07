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
} csf_t;


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
  idx_t const mode,
  splatt_tile_t which_tile);

#define csf_free splatt_csf_free
void csf_free(
  csf_t * const ft);

#endif
