
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "tile.h"
#include "sort.h"
#include "timer.h"
#include "io.h"
#include "ccp/ccp.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Build a pointer structure (i.e. CSR rowptr) into the slabs of tt.
*
* @param inds Indices of just the slice ids.
* @param nnz The number of nonzeros (and thus slice ids).
* @param nslabs The number of slabs to construct.
*
* @return An array of length (nslabs+1) that points into inds and marks the
*         start/end of each slab.
*/
static idx_t * p_mkslabptr(
  idx_t const * const inds,
  idx_t const nnz,
  idx_t const nslabs)
{
  idx_t * slabs = (idx_t *) calloc(nslabs+1, sizeof(idx_t));

  /* make an offset ptr before prefix sum */
  for(idx_t n=0; n < nnz; ++n) {
    idx_t const slabid = inds[n] / TILE_SIZES[0];
    slabs[1 + slabid] += 1;
  }

  for(idx_t s=1; s <= nslabs; ++s) {
    slabs[s] += slabs[s-1];
  }

  return slabs;
}


/**
* @brief Construct a set of unique values (and counts) found within inds.
*
* @param inds The array of indices to tally.
* @param start The first index to tally.
* @param end The last index to tally.
* @param seen An array for marking the counts of each index that is found.
*             NOTE: must at least as large as the largest index.
* @param uniques A sorted array of the unique indices found.
*
* @return The number of unique indices found in ind[start:end].
*/
static idx_t p_fill_uniques(
  idx_t const * const inds,
  idx_t const start,
  idx_t const end,
  idx_t * const seen,
  idx_t * const uniques)
{
  idx_t nuniques = 0;
  for(idx_t n=start; n < end; ++n) {
    idx_t const jj = inds[n];

    /* mark ids and counts of all unique entries */
    seen[jj] += 1;
    if(seen[jj] == 1) {
      uniques[nuniques++] = jj;
    }
  }

  quicksort(uniques, nuniques);
  return nuniques;
}


/**
* @brief Use the uniques/seen arrays to rearrange the nonzeros in a given into
*        a tiled order. Slabs are already ordered after sorting, so this
*        function will be used to first tile into 'tubes' and then finally into
*        proper tiles.
*
* @param start The first nonzero in the working set.
* @param end The last nonzero in the working set.
* @param src The tensor to rearrange.
* @param dest A tensor to write the rearrange slab into.
* @param mode The mode to tile with.
* @param seen An array used to count the number of times each index appears in
*             the mode.
* @param uniques An array used to mark the unique indices. Indexes into seen.
* @param nuniques The number of unique indices in the mode (between start/end).
* @param tsize The dimension of the tiles to construct.
*/
static void p_tile_uniques(
  idx_t const start,
  idx_t const end,
  sptensor_t * const src,
  sptensor_t * const dest,
  idx_t const mode,
  idx_t * const seen,
  idx_t * const uniques,
  idx_t const nuniques,
  idx_t const tsize)
{
  idx_t const ntubes = (nuniques / tsize) + (nuniques % tsize != 0);
  idx_t * tmkr = (idx_t *) calloc(ntubes+1, sizeof(idx_t));

  /* make a marker array so we can quickly move nnz into dest */
  tmkr[0] = start;
  for(idx_t n=0; n < nuniques; ++n) {
    tmkr[1+(n / tsize)] += seen[uniques[n]];
  }
  for(idx_t t=1; t <= ntubes; ++t) {
    tmkr[t] += tmkr[t-1];
  }

  /* reuse seen[] to map ind to unique id */
  for(idx_t n=0; n < nuniques; ++n) {
    seen[uniques[n]] = n;
  }

  /* place nnz */
  idx_t const * const ind = src->ind[mode];
  for(idx_t n=start; n < end; ++n) {
    idx_t const index = tmkr[seen[ind[n]] / tsize];
    for(idx_t m=0; m < src->nmodes; ++m) {
      dest->ind[m][index] = src->ind[m][n];
    }
    dest->vals[index] = src->vals[n];
    tmkr[seen[ind[n]] / tsize] += 1;
  }

  free(tmkr);
}


/**
* @brief Empty a set of unique indices and their counts. Scales with the number
*        of uniques, not the size of the arrays!
*
* @param seen The count for each unique index.
* @param uniques The index of each unique value. Used to index into seen.
* @param nuniques The number of uniques to clear.
*/
static void p_clear_uniques(
  idx_t * const seen,
  idx_t * const uniques,
  idx_t const nuniques)
{
  for(idx_t n=0; n < nuniques; ++n) {
    seen[uniques[n]] = 0;
    uniques[n] = 0;
  }
}


/**
* @brief Rearrange nonzeros into a tiled slab.
*
* @param start The first nonzero in the slab.
* @param end The last nonzero in the slab.
* @param tt The tensor to rearrange.
* @param tt_buf A tensor to use for double-buffering when rearranging.
* @param dim_perm The mode permutation to tile with.
* @param seen An array *for each mode* used to count the number of times each
*             index appears in the slab.
* @param uniques An array *for each mode* used to mark the unique indices. Used
*                to index into seen.
* @param nuniques An idx_t for each mode to count the unique indices in the
*                 slab.
*/
static void p_pack_slab(
  idx_t const start,
  idx_t const end,
  sptensor_t * const tt,
  sptensor_t * const tt_buf,
  idx_t const * const dim_perm,
  idx_t * const * const seen,
  idx_t * const * const uniques,
  idx_t * const nuniques)
{

  idx_t const fibmode = dim_perm[1];
  idx_t const idxmode = dim_perm[2];

  /* get unique fibers */
  nuniques[fibmode] = p_fill_uniques(tt->ind[fibmode], start, end,
    seen[fibmode], uniques[fibmode]);
  p_tile_uniques(start, end, tt, tt_buf, fibmode, seen[fibmode],
    uniques[fibmode], nuniques[fibmode], TILE_SIZES[1]);

  /* get unique idxs */
  nuniques[idxmode] = p_fill_uniques(tt_buf->ind[idxmode], start, end,
    seen[idxmode], uniques[idxmode]);
  p_tile_uniques(start, end, tt_buf, tt, idxmode, seen[idxmode],
    uniques[idxmode], nuniques[idxmode], TILE_SIZES[2]);

  /* Clear out uniques for next slab. Complexity is #uniques, not dimension
   * of tensor... */
  p_clear_uniques(seen[fibmode], uniques[fibmode], nuniques[fibmode]);
  p_clear_uniques(seen[idxmode], uniques[idxmode], nuniques[idxmode]);
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_tile(
  sptensor_t * const tt,
  idx_t * dim_perm)
{
  timer_start(&timers[TIMER_TILE]);
  idx_t const nslices = tt->dims[dim_perm[0]];
  idx_t const nslabs = (nslices / TILE_SIZES[0]) +
    (nslices % TILE_SIZES[0] != 0);

  tt_sort(tt, dim_perm[0], dim_perm);

  sptensor_t * tt_buf = tt_alloc(tt->nnz, tt->nmodes);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    tt_buf->dims[m] = tt->dims[m];
  }

  /* fill in slabs */
  idx_t * slabptr = p_mkslabptr(tt->ind[dim_perm[0]], tt->nnz, nslabs);

  /* seen and uniques are used to mark unique idxs in each slab */
  idx_t * seen[MAX_NMODES];
  idx_t * uniques[MAX_NMODES];
  idx_t nuniques[MAX_NMODES];
  for(idx_t m=1; m < tt->nmodes; ++m) {
    seen[dim_perm[m]]    = (idx_t *) calloc(tt->dims[dim_perm[m]], sizeof(idx_t));
    uniques[dim_perm[m]] = (idx_t *) calloc(tt->dims[dim_perm[m]], sizeof(idx_t));
  }

  /* tile each slab of nonzeros */
  for(idx_t s=0; s < nslabs; ++s) {
    idx_t const start = slabptr[s];
    idx_t const end = slabptr[s+1];

    p_pack_slab(start, end, tt, tt_buf, dim_perm, seen, uniques, nuniques);
  }

  for(idx_t m=1; m < tt->nmodes; ++m) {
    free(seen[dim_perm[m]]);
    free(uniques[dim_perm[m]]);
  }

  tt_free(tt_buf);
  free(slabptr);
  timer_stop(&timers[TIMER_TILE]);
}


idx_t * tt_densetile(
  sptensor_t * const tt,
  idx_t const * const tile_dims)
{
  idx_t const nmodes = tt->nmodes;
  idx_t ntiles = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ntiles *= tile_dims[m];
  }

  /* the actual number of indices to place in each tile */
  idx_t tsizes[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    tsizes[m] = SS_MAX(tt->dims[m] / tile_dims[m], 1);
  }

  idx_t * tcounts = calloc(ntiles+2, sizeof(*tcounts));

  /* count tile sizes (in nnz) */
  idx_t coord[MAX_NMODES];
  for(idx_t x=0; x < tt->nnz; ++x) {
    for(idx_t m=0; m < nmodes; ++m) {
      /* capping at dims-1 fixes overflow when dims don't divide evenly */
      coord[m] = SS_MIN(tt->ind[m][x] / tsizes[m], tile_dims[m]-1);
    }
    /* offset by 1 to make prefix sum easy */
    idx_t const id = get_tile_id(tile_dims, nmodes, coord);
    assert(id < ntiles);
    ++tcounts[2+id];
  }

  /* prefix sum */
  for(idx_t t=3; t <= ntiles+1; ++t) {
    tcounts[t] += tcounts[t-1];
  }

  sptensor_t * newtt = tt_alloc(tt->nnz, tt->nmodes);

  /* copy old tensor into new tiled one */
  for(idx_t x=0; x < tt->nnz; ++x) {
    for(idx_t m=0; m < nmodes; ++m) {
      coord[m] = SS_MIN(tt->ind[m][x] / tsizes[m], tile_dims[m]-1);
    }
    /* offset by 1 to make prefix sum easy */
    idx_t const id = get_tile_id(tile_dims, nmodes, coord);
    assert(id < ntiles);

    idx_t newidx = tcounts[id+1]++;
    newtt->vals[newidx] = tt->vals[x];
    for(idx_t m=0; m < nmodes; ++m) {
      newtt->ind[m][newidx] = tt->ind[m][x];
    }
  }

  /* copy data into old struct */
  memcpy(tt->vals, newtt->vals, tt->nnz * sizeof(val_t));
  for(idx_t m=0; m < nmodes; ++m) {
    memcpy(tt->ind[m], newtt->ind[m], tt->nnz * sizeof(idx_t));
  }
  tt_free(newtt);

  return tcounts;
}


idx_t * tt_ccptile(
  sptensor_t * const tt,
  idx_t const * const tile_dims)
{
  idx_t const nmodes = tt->nmodes;
  idx_t * hist[MAX_NMODES];
  idx_t * parts[MAX_NMODES];
  idx_t ntiles = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ntiles *= tile_dims[m];
    hist[m] = calloc(tt->dims[m], sizeof(**hist));
    parts[m] = splatt_malloc((tile_dims[m]+1) * sizeof(**parts));
  }

  idx_t * tcounts = calloc(ntiles+2, sizeof(*tcounts));

  if(tt->nnz < 10) {
    printf("\nBEFORE\n");
    tt_write(tt, NULL);
  }

  #pragma omp parallel
  {
    /* compute a histogram for each mode */
    for(idx_t m=0; m < nmodes; ++m) {
      idx_t const * const inds = tt->ind[m];
      idx_t * const hgram = hist[m];

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < tt->nnz; ++x) {

        #pragma omp atomic
        ++hgram[inds[x]];
      }
    }
    #pragma omp barrier

    /* now use CCP to partition each mode into tiles */
    #pragma omp for schedule(static, 1)
    for(idx_t m=0; m < nmodes; ++m) {
      partition_1d(hist[m], tt->dims[m], parts[m], tile_dims[m]);
    }

    /* now go back over and count nnz per m-D tile */
    idx_t coord[MAX_NMODES];
    #pragma omp for schedule(dynamic, 32)
    for(idx_t x=0; x < tt->nnz; ++x) {
      /* map nnz to tile coords */
      for(idx_t m=0; m < nmodes; ++m) {
        /* TODO: Binary search. But, tile_dims[m] is usually going to be pretty
         *       small, so this is low priority? */
        for(idx_t p=0; p < tile_dims[m]; ++p) {
          if(parts[m][p+1] > tt->ind[m][x]) {
            coord[m] = p;
            break;
          }
        }
      }
      /* offset by 1 to make prefix sum easy */
      idx_t const id = get_tile_id(tile_dims, nmodes, coord);
      assert(id < ntiles);
      #pragma omp atomic
      ++tcounts[2+id];

    } /* foreach nnz */
  } /* omp parallel */


  /* prefix sum */
  for(idx_t t=3; t <= ntiles+1; ++t) {
    tcounts[t] += tcounts[t-1];
  }

  sptensor_t * newtt = tt_alloc(tt->nnz, tt->nmodes);

  /* copy old tensor into new tiled one */
  idx_t coord[MAX_NMODES];
  for(idx_t x=0; x < tt->nnz; ++x) {
    /* map nnz to linear tile ID, just like before */
    for(idx_t m=0; m < nmodes; ++m) {
      for(idx_t p=0; p < tile_dims[m]; ++p) {
        if(parts[m][p+1] > tt->ind[m][x]) {
          coord[m] = p;
          break;
        }
      }
    }

    /* offset by 1 to make prefix sum easy */
    idx_t const id = get_tile_id(tile_dims, nmodes, coord);
    assert(id < ntiles);

    idx_t newidx = tcounts[id+1]++;
    newtt->vals[newidx] = tt->vals[x];
    for(idx_t m=0; m < nmodes; ++m) {
      newtt->ind[m][newidx] = tt->ind[m][x];
    }
  } /* foreach nnz */

  tt_free(newtt);
  for(idx_t m=0; m < nmodes; ++m) {
    free(hist[m]);
    splatt_free(parts[m]);
  }

  if(tt->nnz < 10) {
    printf("\nAFTER\n");
    tt_write(tt, NULL);
  }

  return tcounts;
}



idx_t get_tile_id(
  idx_t const * const tile_dims,
  idx_t const nmodes,
  idx_t const * const tile_coord)
{
  idx_t id = 0;
  idx_t mult = 1;
  for(idx_t m=nmodes; m-- != 0;) {
    id += tile_coord[m] * mult;
    mult *= tile_dims[m];
  }
  /* bounds check */
  if(id >= mult) {
    id = TILE_ERR;
  }
  return id;
}


void fill_tile_coords(
  idx_t const * const tile_dims,
  idx_t const nmodes,
  idx_t const tile_id,
  idx_t * const tile_coord)
{
  /* Check for invalid id first */
  idx_t maxid = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    maxid *= tile_dims[m];
  }
  if(tile_id >= maxid) {
    for(idx_t m=0; m < nmodes; ++m) {
      tile_coord[m] = tile_dims[m];
    }
    return;
  }

  /* test passed, convert! */
  idx_t id = tile_id;
  for(idx_t m = nmodes; m-- != 0; ) {
    tile_coord[m] = id % tile_dims[m];
    id /= tile_dims[m];
  }
}


idx_t get_next_tileid(
  idx_t const previd,
  idx_t const * const tile_dims,
  idx_t const nmodes,
  idx_t const iter_mode,
  idx_t const mode_idx)
{
  idx_t maxid = 1;
  idx_t coords[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    coords[m] = 0;
    maxid *= tile_dims[m];
  }

  if(previd == TILE_BEGIN) {
    coords[iter_mode] = mode_idx;
    return get_tile_id(tile_dims, nmodes, coords);
  }

  /* check for out of bounds */
  if(previd >= maxid) {
    return TILE_ERR;
  }

  /* convert previd to coords */
  fill_tile_coords(tile_dims, nmodes, previd, coords);

  /* overflowing this mode means TILE_END */
  idx_t const overmode = (iter_mode == 0) ? 1 : 0;

  /* increment least significant mode (unless we're iterating over it) and
   * propagate overflows */
  idx_t pmode = (iter_mode == nmodes-1) ? nmodes-2 : nmodes-1;
  ++coords[pmode];
  while(coords[pmode] == tile_dims[pmode]) {
    if(pmode == overmode) {
      return TILE_END;
    }

    /* overflow this one too and move on */
    coords[pmode] = 0;
    --pmode;

    /* we don't alter the mode we are iterating over */
    if(pmode == iter_mode) {
      /* XXX: checking for overmode should catch this */
      assert(pmode > 0);
      /* if we aren't at the end just skip over it */
      --pmode;
    }

    /* we're now at a valid mode, carry over previous overflow */
    ++coords[pmode];
  }

  return get_tile_id(tile_dims, nmodes, coords);
}



