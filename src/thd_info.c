

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "thd_info.h"
#include <omp.h>


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/
static inline void __reduce_sum(
  thd_info * const thds,
  idx_t const scratchid,
  idx_t const nelems)
{
  int const tid = omp_get_thread_num();
  int const nthreads = omp_get_num_threads();

  val_t * const myvals = (val_t *) thds[tid].scratch[scratchid];

  int half = nthreads / 2;
  while(half > 0) {
    #pragma omp barrier
    if(tid < half && tid + half < nthreads) {
      val_t const * const target = (val_t *) thds[tid+half].scratch[scratchid];
      for(idx_t i=0; i < nelems; ++i) {
        myvals[i] += target[i];
      }
    }
    half /= 2;
  }

  /* account for odd thread at end */
  #pragma omp master
  {
    if(nthreads % 2 == 1) {
      val_t const * const last = (val_t *) thds[nthreads-1].scratch[scratchid];
      for(idx_t i=0; i < nelems; ++i) {
        myvals[i] += last[i];
      }
    }
  }
  #pragma omp barrier
}


static inline void __reduce_max(
  thd_info * const thds,
  idx_t const scratchid,
  idx_t const nelems)
{
  int const tid = omp_get_thread_num();
  int const nthreads = omp_get_num_threads();

  val_t * const myvals = (val_t *) thds[tid].scratch[scratchid];

  int half = nthreads / 2;
  while(half > 0) {
    #pragma omp barrier
    if(tid < half && tid + half < nthreads) {
      val_t const * const target = (val_t *) thds[tid+half].scratch[scratchid];
      for(idx_t i=0; i < nelems; ++i) {
        myvals[i] = SS_MAX(myvals[i], target[i]);
      }
    }
    half /= 2;
  }

  /* account for odd thread at end */
  #pragma omp master
  {
    if(nthreads % 2 == 1) {
      val_t const * const last = (val_t *) thds[nthreads-1].scratch[scratchid];
      for(idx_t i=0; i < nelems; ++i) {
        myvals[i] = SS_MAX(myvals[i], last[i]);
      }
    }
  }
  #pragma omp barrier
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void thd_reduce(
  thd_info * const thds,
  idx_t const scratchid,
  idx_t const nelems,
  splatt_reduce_type const which)
{
  switch(which) {
  case REDUCE_SUM:
    __reduce_sum(thds, scratchid, nelems);
    break;
  case REDUCE_MAX:
    __reduce_max(thds, scratchid, nelems);
    break;
  default:
    fprintf(stderr, "SPLATT: thd_reduce supports SUM and MAX only.\n");
    abort();
  }
}


thd_info * thd_init(
  idx_t const nthreads,
  idx_t const nscratch,
  ...)
{
  thd_info * thds = (thd_info *) malloc(nthreads * sizeof(thd_info));

  for(idx_t t=0; t < nthreads; ++t) {
    timer_reset(&thds[t].ttime);
    thds[t].nscratch = nscratch;
    thds[t].scratch = (void **) malloc(nscratch * sizeof(void*));
  }

  va_list args;
  va_start(args, nscratch);
  for(idx_t s=0; s < nscratch; ++s) {
    idx_t const bytes = va_arg(args, idx_t);
    for(idx_t t=0; t < nthreads; ++t) {
      thds[t].scratch[s] = (void *) malloc(bytes);
    }
  }
  va_end(args);

  return thds;
}

void thd_times(
  thd_info * thds,
  idx_t const nthreads)
{
  for(idx_t t=0; t < nthreads; ++t) {
    printf("  thd: %"SS_IDX" %0.3fs\n", t, thds[t].ttime.seconds);
  }
}

void thd_reset(
  thd_info * thds,
  idx_t const nthreads)
{
  for(idx_t t=0; t < nthreads; ++t) {
    timer_reset(&thds[t].ttime);
  }
}

void thd_free(
  thd_info * thds,
  idx_t const nthreads)
{
  for(idx_t t=0; t < nthreads; ++t) {
    for(idx_t s=0; s < thds[t].nscratch; ++s) {
      free(thds[t].scratch[s]);
    }
    free(thds[t].scratch);
  }
  free(thds);
}

