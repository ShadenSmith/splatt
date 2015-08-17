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


/*
 * Constants used during tile traversal.
 */
static idx_t const TILE_ERR   = SPLATT_IDX_MAX - 2;
static idx_t const TILE_BEGIN = SPLATT_IDX_MAX - 1;
static idx_t const TILE_END   = SPLATT_IDX_MAX;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


/******************************************************************************
 * TILE TRAVERSAL
 *****************************************************************************/

#define get_tile_id splatt_get_tile_id
/**
* @brief Convert the cartesian coordinates of a tile into its linear ID.
*
* @param tile_dims The dimensions of the tile system.
* @param nmodes The number of modes in the system.
* @param tile_coord The coordinates of the tile.
*
* @return The ID of the tile, or TILE_ERR if out of bounds.
*/
idx_t get_tile_id(
  idx_t const * const tile_dims,
  idx_t const nmodes,
  idx_t const * const tile_coord);


#define fill_tile_coords splatt_fill_tile_coords
/**
* @brief Fill 'tile_coord' with the coordinates of a specified tile ID.
*
* @param tile_dims The dimensions of the tile syst.m
* @param nmodes The number of modes in the system.
* @param tile_id The id to convert.
* @param tile_coord [OUT] The coordinates of the tile ID. Returns
*                   tile_dims[m] for any invalid IDs.
*/
void fill_tile_coords(
  idx_t const * const tile_dims,
  idx_t const nmodes,
  idx_t const tile_id,
  idx_t * const tile_coord);


#define get_next_tileid splatt_get_next_tileid
/**
* @brief Find the ID of the next tile in a traversal. If a thread is iterating
*        over all tiles T(:,1,:), this function can be looped over to handle
*        all tile ID indexing and offsets. Sample usage:
*
*        id = get_next_tileid(TILE_BEGIN, dims, tt->nmodes, mode, thread_id);
*        while(id != TILE_END) {
*          <do processing>
*          id = get_next_tileid(id, dims, tt->nmodes, mode, thread_id);
*        }
*
* @param previd The previous ID that was processed. Start with TILE_BEGIN.
* @param tile_dims The dimensionality of each mode of tiling.
* @param nmodes The number of modes in the tile system.
* @param mode The mode we are iterating over.
* @param mode_idx The index of the mode, (e.g. T(:,1,:) would require mode=1,
*                 mode_idx=1).
*
* @return  The ID of the next tile to process. TILE_END will be returned if
*          if all tiles are processed in T(:, mode_idx, :). TILE_ERR will be
*          returned if previd is out of bounds.
*/
idx_t get_next_tileid(
  idx_t const previd,
  idx_t const * const tile_dims,
  idx_t const nmodes,
  idx_t const mode,
  idx_t const mode_idx);


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
