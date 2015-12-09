
#include <stdio.h>
#include <stdlib.h>

#include "../include/splatt.h"

int main(int argc, char ** argv)
{
  if(argc < 3) {
    printf("usage: %s <tensor> <rank> [THREADS ...]\n", *argv);
    return 1;
  }

  int ret;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE;
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  opts[SPLATT_OPTION_VERBOSITY] = SPLATT_VERBOSITY_MAX;
  opts[SPLATT_OPTION_TOLERANCE] = 0.;
  opts[SPLATT_OPTION_NITER] = 10;

  splatt_idx_t nmodes;
  splatt_csf * csf;
  ret = splatt_csf_load(argv[1], &nmodes, &csf, opts);
  if(ret != SPLATT_SUCCESS) {
    printf("splatt_csf_load: %d\n", ret);
    return 1;
  }

  splatt_idx_t * rank = malloc(nmodes * sizeof(*rank));
  for(splatt_idx_t m=0; m < nmodes; ++m) {
    rank[m] = (splatt_idx_t) atoi(argv[2]);
  }

  splatt_timer_inc_verbose();
  splatt_timer_inc_verbose();
  splatt_timer_inc_verbose();

  /* tucker */
  int i;
  for(i=3; i < argc; ++i) {
    int nthreads = atoi(argv[i]);
    opts[SPLATT_OPTION_NTHREADS] = nthreads;

    printf("\n\n");

    splatt_init_timers();

    splatt_tucker_t factored;
    ret = splatt_tucker_als(rank, nmodes, csf, opts, &factored);

    splatt_report_times();

    splatt_free_tucker(&factored);
  }

  free(rank);
  splatt_free_csf(csf, opts);
  splatt_free_opts(opts);

  return 0;
}
