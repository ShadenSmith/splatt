
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "tile.h"
#include "sort.h"
#include "timer.h"
#include "io.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/
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


static idx_t __fill_uniques(
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


static void __tile_uniques(
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


static void __clear_uniques(
  idx_t * const seen,
  idx_t * const uniques,
  idx_t const nuniques)
{
  for(idx_t n=0; n < nuniques; ++n) {
    seen[uniques[n]] = 0;
    uniques[n] = 0;
  }
}


static void __pack_slab(
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
  nuniques[fibmode] = __fill_uniques(tt->ind[fibmode], start, end,
    seen[fibmode], uniques[fibmode]);
  __tile_uniques(start, end, tt, tt_buf, fibmode, seen[fibmode],
    uniques[fibmode], nuniques[fibmode], TILE_SIZES[1]);

  /* get unique idxs */
  nuniques[idxmode] = __fill_uniques(tt_buf->ind[idxmode], start, end,
    seen[idxmode], uniques[idxmode]);
  __tile_uniques(start, end, tt_buf, tt, idxmode, seen[idxmode],
    uniques[idxmode], nuniques[idxmode], TILE_SIZES[2]);

  /* Clear out uniques for next slab. Complexity is #uniques, not dimension
   * of tensor... */
  __clear_uniques(seen[fibmode], uniques[fibmode], nuniques[fibmode]);
  __clear_uniques(seen[idxmode], uniques[idxmode], nuniques[idxmode]);
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
  idx_t * slabptr = __mkslabptr(tt->ind[dim_perm[0]], tt->nnz, nslabs);

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

    __pack_slab(start, end, tt, tt_buf, dim_perm, seen, uniques, nuniques);
  }

  for(idx_t m=1; m < tt->nmodes; ++m) {
    free(seen[dim_perm[m]]);
    free(uniques[dim_perm[m]]);
  }

  tt_free(tt_buf);
  free(slabptr);
  timer_stop(&timers[TIMER_TILE]);
}


