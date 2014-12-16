
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "tile.h"
#include "sort.h"
#include "timer.h"

static idx_t * __mkslabptr(
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

static void __lol(
  idx_t const * const inds,
  idx_t const start,
  idx_t const end,
  idx_t * const seen)
{
  idx_t * uniques;
  idx_t nuniques;
  for(idx_t n=start; n < end; ++n) {
    idx_t const jj = inds[n];

    /* mark ids and counts of all unique entries */
    seen[jj] += 1;
    if(seen[jj] == 1) {
      uniques[nuniques++] = jj;
    }
  }

  quicksort(uniques, nuniques);
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
  printf("nslices: "SS_IDX"  nslabs: " SS_IDX"\n", nslices, nslabs);

  tt_sort(tt, dim_perm[0], dim_perm);

  /* fill in slabs */
  idx_t * slabptr = __mkslabptr(tt->ind[dim_perm[0]], tt->nnz, nslabs);
  idx_t * fibptr = (idx_t *) calloc(tt->dims[dim_perm[1]], sizeof(idx_t));

  for(idx_t s=0; s < nslabs; ++s) {
    idx_t const start = slabptr[s];
    idx_t const end = slabptr[s+1];
  }

  free(fibptr);
  free(slabptr);
  timer_stop(&timers[TIMER_TILE]);
}


