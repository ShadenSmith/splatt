
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "matrix.h"
#include "sptensor.h"
#include "stats.h"
#include "timer.h"
#include "thd_info.h"
#include "tile.h"
#include "io.h"
#include "util.h"
#include "ttm.h"

#include <omp.h>

/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
int splatt_tucker_als(
    splatt_idx_t const * const nfactors,
    splatt_idx_t const nmodes,
    splatt_csf const * const tensors,
    double const * const options,
    splatt_tucker_t * factored)
{
  matrix_t * mats[MAX_NMODES+1];

  idx_t const nthreads = (idx_t) options[SPLATT_OPTION_NTHREADS];

  /* fill in factored */
  idx_t maxcols = 0;
  idx_t csize = 1;
  factored->nmodes = nmodes;
  for(idx_t m=0; m < nmodes; ++m) {
    factored->rank[m] = nfactors[m];
    mats[m] = mat_rand(tensors[0].dims[m], nfactors[m]);
    factored->factors[m] = mats[m]->vals;

    csize *= nfactors[m];
    maxcols = SS_MAX(maxcols, nfactors[m]);
  }
  factored->core = (val_t *) calloc(csize, sizeof(val_t));

  idx_t const tenout_size = tenout_dim(nmodes, nfactors, tensors->dims);
  val_t * gten = malloc(tenout_size * sizeof(*gten));

  /* thread structures */
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 1,
    (maxcols * sizeof(val_t)) + 64);

  sp_timer_t itertime;
  sp_timer_t modetime[MAX_NMODES];

  double oldfit = 0;
  double fit = 0;

  /* foreach iteration */
  idx_t const niters = (idx_t) options[SPLATT_OPTION_NITER];
  for(idx_t it=0; it < niters; ++it) {
    timer_fstart(&itertime);

    /* foreach mode */
    for(idx_t m=0; m < nmodes; ++m) {
      timer_fstart(&modetime[m]);

      timer_start(&timers[TIMER_TTM]);
      ttm_csf(tensors, mats, gten, m, thds, options);
      timer_stop(&timers[TIMER_TTM]);

      timer_stop(&modetime[m]);
    }

    timer_stop(&itertime);

    /* print progress */
    if(options[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_NONE) {
      printf("  its = %3"SPLATT_PF_IDX" (%0.3fs)  fit = %0.5f  delta = %+0.4e\n",
          it+1, itertime.seconds, fit, fit - oldfit);
      if(options[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_LOW) {
        for(idx_t m=0; m < nmodes; ++m) {
          printf("     mode = %1"SPLATT_PF_IDX" (%0.3fs)\n", m+1,
              modetime[m].seconds);
        }
      }
    }
  }

  /* cleanup */
  free(gten);
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]);
  }
  return SPLATT_SUCCESS;
}


void splatt_free_tucker(
    splatt_tucker_t * factored)
{
  free(factored->core);
  for(idx_t m=0; m < factored->nmodes; ++m) {
    free(factored->factors[m]);
  }
}


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

