#ifndef SPLATT_TIMER_H
#define SPLATT_TIMER_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#define _GNU_SOURCE
#include <time.h>



/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  int running;
  double seconds;
  struct timespec start;
  struct timespec stop;
} sp_timer_t;

typedef enum
{
  TIMER_ALL,
  TIMER_IO,
  TIMER_SPLATT,
  TIMER_GIGA,
  TIMER_DFACTO,
  TIMER_TTBOX,
  TIMER_SORT,
  TIMER_MISC,
  TIMER_REORDER,
  TIMER_NTIMERS
} timer_id;

extern sp_timer_t timers[];

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void init_timers(void);
void report_times(void);

static inline void timer_reset(sp_timer_t * const timer)
{
  timer->running       = 0;
  timer->seconds       = 0;
  timer->start.tv_sec  = 0;
  timer->start.tv_nsec = 0;
  timer->stop.tv_sec   = 0;
  timer->stop.tv_nsec  = 0;
}

static inline void timer_start(sp_timer_t * const timer)
{
  timer->running = 1;
  clock_gettime(CLOCK_MONOTONIC, &(timer->start));
}

static inline void timer_stop(sp_timer_t * const timer)
{
  clock_gettime(CLOCK_MONOTONIC, &(timer->stop));
  timer->running = 0;
  timer->seconds += (double)(timer->stop.tv_sec - timer->start.tv_sec);
  timer->seconds += (timer->stop.tv_nsec - timer->start.tv_nsec)*1e-9;
}

static inline void timer_fstart(sp_timer_t * const timer)
{
  timer_reset(timer);
  timer_start(timer);
}

#endif
