

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "timer.h"
#include <stdio.h>


/******************************************************************************
 * PRIVATE STRUCTURES
 *****************************************************************************/
static char const * const timer_names[] = {
  [TIMER_ALL]     = "TOTAL",
  [TIMER_IO]      = "IO",
  [TIMER_SPLATT]  = "SPLATT",
  [TIMER_GIGA]    = "GIGA",
  [TIMER_TTBOX]   = "TTBOX",
  [TIMER_DFACTO]  = "DFACTO",
  [TIMER_REORDER] = "REORDER",
  [TIMER_SORT]    = "SORT",
  [TIMER_TILE]    = "TILE",
  [TIMER_MISC]    = "MISC"
};

/* definition of global timers[] */
sp_timer_t timers[TIMER_NTIMERS];



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void init_timers(void)
{
  for(int t=0; t < TIMER_NTIMERS; ++t) {
    timer_reset(&timers[t]);
  }
}

void report_times(void)
{
  printf("\n");
  printf("Timing information ---------------------------------------------\n");
  for(int t=0; t < TIMER_NTIMERS; ++t) {
    if(timers[t].seconds > 0) {
      printf("  %-20s%0.3fs\n", timer_names[t], timers[t].seconds);
    }
  }
}

