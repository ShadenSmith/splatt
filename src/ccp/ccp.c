

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "ccp.h"
#include "../timer.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static idx_t nprobes = 0;


/**
* @brief Perform a linear search on an array for a value.
*
* @param weights The array to search.
* @param left The lower bound to begin at.
* @param right The upper (exclusive) bound of items.
* @param target The target value.
*
* @return The index j, where weights[j] <= target && weights[j+1] > target.
*/
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


/**
* @brief Perform a binary search on an array for a value.
*
* @param weights The array to search.
* @param left The lower bound to begin at.
* @param right The upper (exclusive) bound of items.
* @param target The target value.
*
* @return The index j, where weights[j] <= target && weights[j+1] > target.
*/
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



static idx_t p_eps_rb_partition_1d(
    idx_t * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts,
    idx_t const eps)
{
  idx_t lower = weights[nitems-1] / nparts;
  idx_t upper = weights[nitems-1];

  do {
    idx_t mid = lower + ((upper - lower) / 2);
    if(lprobe(weights, nitems, parts, nparts, mid)) {
      upper = mid;
    } else {
      lower = mid+1;
    }
  } while(upper > lower + eps);

  return upper;
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
  timer_start(&timers[TIMER_PART]);
  prefix_sum_inc(weights, nitems);

  nprobes = 0;

  /* use recursive bisectioning with 0 tolerance to get exact solution */
  idx_t bottleneck = p_eps_rb_partition_1d(weights, nitems, parts, nparts, 0);

  printf("nprobes: %lu\n", nprobes);

  timer_stop(&timers[TIMER_PART]);
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

  parts[p] = nitems;

  ++nprobes;
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
