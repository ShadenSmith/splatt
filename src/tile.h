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


static idx_t const TILE_BEGIN = SPLATT_IDX_MAX - 1;
static idx_t const TILE_END   = SPLATT_IDX_MAX;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

idx_t get_tile_id(
  idx_t const * const tile_dims,
  idx_t const nmodes,
  idx_t const * const tile_coord);


void fill_tile_coords(
  idx_t const * const tile_dims,
  idx_t const nmodes,
  idx_t const tile_id,
  idx_t * const tile_coord);


idx_t get_next_tileid(
  idx_t const previd,
  idx_t const * const tile_dims,
  idx_t const nmodes,
  idx_t const mode_traversed);


void tt_threadtile(
  sptensor_t * const tt,
  idx_t const nthreads);


#define tt_tile splatt_tt_tile
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
