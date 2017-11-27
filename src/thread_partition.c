


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "thread_partition.h"
#include "timer.h"


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
    if(target < weights[x+1]) {
      return x+1;
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


/*
 * Not static because we use it in unit tests.
 */
bool lprobe(
    idx_t const * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts,
    idx_t const bottleneck)
{
  ++nprobes;
  idx_t const wtotal = weights[nitems-1];
  /* initialize partitioning */
  parts[0] = 0;
  for(idx_t p=1; p <= nparts; ++p) {
    parts[p] = nitems;
  }

  idx_t bsum = bottleneck;
  idx_t step = nitems / nparts;
  for(idx_t p=1; p < nparts; ++p) {
    /* jump to the next bucket */
    while(step < nitems && weights[step] < bsum) {
      step += nitems / nparts;
    }

    /* find the end (exclusive) index of process p */
    parts[p] = p_binary_search(weights, step - (nitems/nparts),
        SS_MIN(step,nitems), bsum);

    /* we ran out of stuff to do */
    if(parts[p] == nitems) {
      /* check for pathological case when the last weight is larger than
       * bottleneck */
      idx_t const size_last = weights[nitems-1] - weights[parts[p-1]-1];
      return size_last < bottleneck;
    }
    bsum = weights[parts[p]-1] + bottleneck;
  }

  return bsum >= wtotal;
}


static idx_t p_eps_rb_partition_1d(
    idx_t * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts,
    idx_t const eps)
{
  idx_t const tot_weight = weights[nitems-1];
  idx_t lower = tot_weight / nparts;
  idx_t upper = tot_weight;

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



idx_t * partition_weighted(
    idx_t * const weights,
    idx_t const nitems,
    idx_t const nparts,
    idx_t * const bottleneck)
{
  timer_start(&timers[TIMER_PART]);
  prefix_sum_inc(weights, nitems);

  idx_t * parts = splatt_malloc((nparts+1) * sizeof(*parts));

  nprobes = 0;

  idx_t bneck = 0;
  
  /* actual partitioning */
  if(nitems > nparts) {
    /* use recursive bisectioning with 0 tolerance to get exact solution */
    bneck = p_eps_rb_partition_1d(weights, nitems, parts, nparts, 0);
    /* apply partitioning that we found */
    bool success = lprobe(weights, nitems, parts, nparts, bneck);
    assert(success == true);

  /* Do a trivial partitioning. Silly, but this happens when tensors have
   * short modes. */
  } else {
    for(idx_t p=0; p < nitems; ++p) {
      parts[p] = p;
      bneck = SS_MAX(bneck, weights[p]);
    }
    for(idx_t p=nitems; p <= nparts; ++p) {
      parts[p] = nitems;
    }
  }

  *bottleneck = bneck;

  timer_stop(&timers[TIMER_PART]);
  return parts;
}


idx_t * partition_simple(
    idx_t const nitems,
    idx_t const nparts)
{
  timer_start(&timers[TIMER_PART]);

  idx_t * parts = splatt_malloc((nparts+1) * sizeof(*parts));

  parts[0] = 0;
  idx_t const per_part = SS_MAX(nitems / nparts, 1);
  for(idx_t p=1; p < nparts; ++p) {
    parts[p] = SS_MAX(SS_MIN(per_part * p, nitems), 1);
  }
  parts[nparts] = nitems;

  timer_stop(&timers[TIMER_PART]);
  return parts;
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
