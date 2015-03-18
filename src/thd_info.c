

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "thd_info.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
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

