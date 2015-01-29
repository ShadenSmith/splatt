
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "mttkrp.h"
#include "timer.h"
#include "thd_info.h"
#include "matrix.h"
#include "tile.h"
#include "io.h"

#include <omp.h>


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void cpd(
  sptensor_t * const tt,
  matrix_t ** mats,
  cpd_opts const * const opts)
{
  idx_t const rank = opts->rank;
  idx_t const nmodes = tt->nmodes;

  val_t * lambda = (val_t *) malloc(rank * sizeof(val_t));
  matrix_t * ata    = mat_alloc(rank, rank);
  matrix_t * atabuf = mat_alloc(rank, rank);
  ftensor_t * ft = ften_alloc(tt, opts->tile);

  /* setup thread structures */
  omp_set_num_threads(opts->nthreads);
  thd_info * thds;
  if(opts->tile) {
    thds = thd_init(opts->nthreads, rank * sizeof(val_t) + 64,
      TILE_SIZES[0] * rank * sizeof(val_t) + 64);
  } else {
    thds = thd_init(opts->nthreads, rank * sizeof(val_t) + 64, 0);
  }

  /* setup timers */
  sp_timer_t itertime;
  timer_reset(&timers[TIMER_SPLATT]);

  for(idx_t it=0; it < opts->niters; ++it) {
    timer_fstart(&itertime);
    for(idx_t m=0; m < nmodes; ++m) {
      /* M1 = X * (C o B) */
      timer_start(&timers[TIMER_SPLATT]);
      mttkrp_splatt(ft, mats, m, thds, opts->nthreads);
      timer_stop(&timers[TIMER_SPLATT]);

      /* normalize columns and extract lambda if necessary */
      if(it == 0) {
        mat_normalize(mats[m], lambda, MAT_NORM_2);
      } else {
        mat_normalize(mats[m], lambda, MAT_NORM_MAX);
      }

      /* M2 = (CtC * BtB * ...) */
      mat_aTa_hada(mats, (m+1) % nmodes, m, nmodes, atabuf, ata);

      /* M2 = LL^T */
      mat_syminv(ata, atabuf);
    }
    timer_stop(&itertime);
    printf("    its = " SS_IDX " (%0.3fs)\n", it+1, itertime.seconds);
  }

  ften_free(ft);
  thd_free(thds, opts->nthreads);
  mat_free(ata);
  mat_free(atabuf);
  free(lambda);
}
