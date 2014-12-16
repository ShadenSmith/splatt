#ifndef SPLATT_TILE_H
#define SPLATT_TILE_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"

#define SLAB_TILE 8
#define FIB_TILE 8



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_tile(
  sptensor_t * const tt,
  idx_t * dim_perm);

#endif
