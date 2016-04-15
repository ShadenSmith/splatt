
#include "base.h"
#include <stdlib.h>
#include <omp.h>


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
double * splatt_default_opts(void)
{
  double * opts = (double *) splatt_malloc(SPLATT_OPTION_NOPTIONS * \
      sizeof(*opts));
  for(int i=0; i < SPLATT_OPTION_NOPTIONS; ++i) {
    opts[i] = SPLATT_VAL_OFF;
  }

  opts[SPLATT_OPTION_TOLERANCE]  = DEFAULT_TOL;
  opts[SPLATT_OPTION_REGULARIZE] = 0.;;
  opts[SPLATT_OPTION_NITER]      = DEFAULT_ITS;
  opts[SPLATT_OPTION_VERBOSITY]  = SPLATT_VERBOSITY_LOW;

  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_TWOMODE;
  opts[SPLATT_OPTION_TILE]      = SPLATT_NOTILE;
  opts[SPLATT_OPTION_TILEDEPTH] = 1;

  opts[SPLATT_OPTION_DECOMP] = SPLATT_DECOMP_MEDIUM;
  opts[SPLATT_OPTION_COMM]   = SPLATT_COMM_ALL2ALL;

  if(omp_in_parallel()) {
    opts[SPLATT_OPTION_NTHREADS]  = 1;
  } else {
    opts[SPLATT_OPTION_NTHREADS]  = omp_get_num_procs();
  }


  return opts;
}

void splatt_free_opts(
  double * opts)
{
  free(opts);
}

