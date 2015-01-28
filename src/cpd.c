
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "mttkrp.h"
#include "timer.h"
#include "thd_info.h"
#include "matrix.h"


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

  sp_timer_t ata_time;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    timer_fstart(&ata_time);
    mat_aTa(mats[m], ata);
    timer_stop(&ata_time);
    printf("time: %0.4fs\n", ata_time.seconds);
  }

  mat_free(ata);
}
