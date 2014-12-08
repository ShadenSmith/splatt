#ifndef SP_TIMER_H
#define SP_TIMER_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
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



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
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

#endif
