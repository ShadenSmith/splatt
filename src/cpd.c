
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
  matrix_t * ata = mat_alloc(rank, rank);
  matrix_t * btb = mat_alloc(rank, rank);
  ftensor_t * ft = ften_alloc(tt, opts->tile);
  thd_info * thds;
  if(opts->tile) {
    thds = thd_init(opts->nthreads, rank * sizeof(val_t) + 64,
      TILE_SIZES[0] * rank * sizeof(val_t) + 64);
  } else {
    thds = thd_init(opts->nthreads, rank * sizeof(val_t) + 64, 0);
  }

  omp_set_num_threads(opts->nthreads);

  for(idx_t it=0; it < opts->niters; ++it) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      mttkrp_splatt(ft, mats, m, thds, opts->nthreads);
    }
  }

  sp_timer_t ata_time;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    timer_fstart(&ata_time);
    mat_aTa(mats[m], ata);
    timer_stop(&ata_time);
    printf("time: %0.4fs\n", ata_time.seconds);
  }

  ften_free(ft);
  thd_free(thds, opts->nthreads);
  mat_free(ata);
  mat_free(btb);
}
