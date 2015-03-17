

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "thd_info.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
thd_info * thd_init(
  idx_t const nthreads,
  idx_t const scratch1_bytes,
  idx_t const scratch2_bytes)
{
  thd_info * thds = (thd_info *) malloc(nthreads * sizeof(thd_info));

  for(idx_t t=0; t < nthreads; ++t) {
    timer_reset(&thds[t].ttime);
    thds[t].scratch  = (void *) malloc(scratch1_bytes);
    thds[t].scratch2 = (void *) malloc(scratch2_bytes);
  }
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
    free(thds[t].scratch);
    free(thds[t].scratch2);
  }
  free(thds);
}

