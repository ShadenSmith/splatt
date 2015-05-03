#ifndef SPLATT_TILE_H
#define SPLATT_TILE_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"

/** Dimensions to use while tiling
 *  TODO: make this configurable? */
static idx_t const TILE_SIZES[] = { 2048, 2048, 2048 };


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Rearrange the nonzeros of a tensor into a tiled form.
*
* @param tt The tensor to tile.
* @param dim_perm Defines the mode permutation to tile over.
*                 mode dim_perm[0] defines 'slabs' of the tiles which are of
*                 size TILE_SIZES[0]. Fibers are along mode dim_perm[2] and
*                 TILE_SIZES[2] defines the longest fibers allowed in a tile.
*/
void tt_tile(
  sptensor_t * const tt,
  idx_t * dim_perm);

#endif
