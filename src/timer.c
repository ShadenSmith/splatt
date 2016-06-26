

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "timer.h"
#include <stdio.h>


/******************************************************************************
 * PRIVATE STRUCTURES
 *****************************************************************************/
static char const * const timer_names[] = {
  [TIMER_ALL]       = "TOTAL",
  [TIMER_CPD]       = "CPD",
  [TIMER_IO]        = "IO",
  [TIMER_MTTKRP]    = "MTTKRP",
  [TIMER_INV]       = "INVERSE",
  [TIMER_SPLATT]    = "SPLATT",
  [TIMER_GIGA]      = "GIGA",
  [TIMER_TTBOX]     = "TTBOX",
  [TIMER_DFACTO]    = "DFACTO",
  [TIMER_REORDER]   = "REORDER",
  [TIMER_SORT]      = "SORT",
  [TIMER_TILE]      = "TILE",
  [TIMER_CONVERT]   = "CONVERT",
  [TIMER_FIT]       = "CPD FIT",
  [TIMER_MATMUL]    = "MAT MULT",
  [TIMER_ATA]       = "MAT A^TA",
  [TIMER_MATNORM]   = "MAT NORM",
  [TIMER_PART]      = "PART1D",
  [TIMER_MISC]      = "MISC",
#ifdef SPLATT_USE_MPI
  [TIMER_MPI]           = "MPI",
  [TIMER_MPI_IDLE]      = "MPI IDLE",
  [TIMER_MPI_COMM]      = "MPI COMM",
  [TIMER_MPI_ATA]       = "MPI ATA",
  [TIMER_MPI_REDUCE]    = "MPI RED",
  [TIMER_MPI_PARTIALS]  = "MPI PARTS",
  [TIMER_MPI_NORM]      = "MPI NORM",
  [TIMER_MPI_UPDATE]    = "MPI UPD",
  [TIMER_MPI_FIT]       = "MPI FIT",
  [TIMER_MTTKRP_MAX]    = "MTTKRP MAX",
  [TIMER_MPI_MAX]       = "MPI MAX",
  [TIMER_MPI_IDLE_MAX]  = "MPI IDLE MAX",
  [TIMER_MPI_COMM_MAX]  = "MPI COMM MAX",
#endif
};


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void init_timers(void)
{
  timer_lvl = TIMER_LVL1;
  for(int t=0; t < TIMER_NTIMERS; ++t) {
    timer_reset(&timers[t]);
  }
}

void report_times(void)
{
  printf("\n");
  printf("Timing information ---------------------------------------------\n");
  for(int t=0; t < timer_lvl; ++t) {
    if(timers[t].seconds > 0) {
      printf("  %-20s%0.3fs\n", timer_names[t], timers[t].seconds);
    }
  }
}

void timer_inc_verbose(void)
{
  switch(timer_lvl) {
  case TIMER_LVL0:
    timer_lvl = TIMER_LVL1;
    break;
  case TIMER_LVL1:
    timer_lvl = TIMER_LVL2;
    break;
  case TIMER_LVL2:
    timer_lvl = TIMER_NTIMERS;
    break;
  default:
    break;
  }
}

