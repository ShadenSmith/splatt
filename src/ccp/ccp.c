

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "ccp.h"

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


static idx_t p_linear_search(
    idx_t const * const weights,
    idx_t const left,
    idx_t const right,
    idx_t const target)
{
  for(idx_t x=left; x < right-1; ++x) {
    if(weights[x+1] > target) {
      return x;
    }
  }

  return right;
}

static idx_t p_binary_search(
    idx_t const * const weights,
    idx_t left,
    idx_t right,
    idx_t const target)
{
  while((right - left) > 8) {
    idx_t mid = left + ((right - left) / 2);

    if(weights[mid] <= target && weights[mid+1] > target) {
      return mid;
    }

    if(weights[mid] < target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  return p_linear_search(weights, left, right, target);
}




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

idx_t partition_1d(
    idx_t * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts)
{
  prefix_sum_inc(weights, nitems);

  idx_t const total_weight = weights[nitems-1];

  idx_t nprobes = 0;

  /* naive attempts */
  bool success;
  idx_t bottleneck = (total_weight / nparts) - 1; /* -1 because we inc first */
  do {
    ++nprobes;
    ++bottleneck;
    success = lprobe(weights, nitems, parts, nparts, bottleneck);
  } while(!success);

  printf("nprobes: %lu\n", nprobes);

  return bottleneck;
}



bool lprobe(
    idx_t const * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts,
    idx_t const bottleneck)
{
  idx_t p=0;
  parts[p++] = 0;
  idx_t bsum = bottleneck;

  idx_t const wtotal = weights[nitems-1];

#if 0
  while(p < nparts && bsum < wtotal) {
    parts[p] = p_linear_search(weights, parts[p-1], nitems, bsum);
    bsum = weights[parts[p]] + bottleneck;
    ++p;
  }
#else
  idx_t step = nitems / nparts;
  while(p < nparts && bsum < wtotal) {
    while(step < nitems && weights[step] < bsum) {
      step += nitems / nparts;
    }
    parts[p] = p_binary_search(weights, step - (nitems/nparts), SS_MIN(step, nitems),
        bsum);
    bsum = weights[parts[p]] + bottleneck;
    ++p;
  }
#endif

  parts[p] = nitems;

  return bsum >= wtotal;
}


void prefix_sum_inc(
    idx_t * const weights,
    idx_t const nitems)
{
  for(idx_t x=1; x < nitems; ++x) {
    weights[x] += weights[x-1];
  }
}



void prefix_sum_exc(
    idx_t * const weights,
    idx_t const nitems)
{
  idx_t saved = weights[0];
  weights[0] = 0;
  for(idx_t x=1; x < nitems; ++x) {
    idx_t const tmp = weights[x];
    weights[x] = weights[x-1] + saved;
    saved = tmp;
  }
}
