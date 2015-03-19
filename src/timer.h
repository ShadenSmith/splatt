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

/**
* @brief Represents a wall-clock timer.
*/
typedef struct
{
  int running;
  double seconds;
  struct timespec start;
  struct timespec stop;
} sp_timer_t;


/**
* @brief timer_id provides easy indexing into timers[].
*/
typedef enum
{
  TIMER_LVL0,   /* LEVEL 0 */
  TIMER_ALL,
  TIMER_LVL1,   /* LEVEL 1 */
  TIMER_IO,
  /* COMMANDS */
  TIMER_CPD,
  TIMER_REORDER,
  TIMER_CONVERT,
  TIMER_LVL2,   /* LEVEL 2 */
  TIMER_MTTKRP,
  TIMER_INV,
  TIMER_FIT,
  TIMER_MATMUL,
  TIMER_ATA,
  TIMER_MATNORM,
#ifdef USE_MPI
  TIMER_MPI,
  TIMER_MPI_IDLE,
  TIMER_MPI_COMM,
  /* timer max */
  TIMER_MTTKRP_MAX,
  TIMER_MPI_MAX,
  TIMER_MPI_IDLE_MAX,
  TIMER_MPI_COMM_MAX,
#endif
  TIMER_SPLATT,
  TIMER_GIGA,
  TIMER_DFACTO,
  TIMER_TTBOX,
  TIMER_SORT,
  TIMER_TILE,
  TIMER_MISC,
  TIMER_NTIMERS /* LEVEL N */
} timer_id;


static int timer_lvl = TIMER_NTIMERS;
extern sp_timer_t timers[];

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


/**
* @brief Call timer_reset() on all of timers[].
*/
void init_timers(void);


/**
* @brief Output a summary of all used timers.
*/
void report_times(void);


/**
* @brief Reset all fields of a sp_timer_t.
*
* @param timer The timer to reset.
*/
static inline void timer_reset(sp_timer_t * const timer)
{
  timer->running       = 0;
  timer->seconds       = 0;
  timer->start.tv_sec  = 0;
  timer->start.tv_nsec = 0;
  timer->stop.tv_sec   = 0;
  timer->stop.tv_nsec  = 0;
}


/**
* @brief Start a sp_timer_t. NOTE: this does not reset the timer.
*
* @param timer The timer to start.
*/
static inline void timer_start(sp_timer_t * const timer)
{
  timer->running = 1;
  clock_gettime(CLOCK_MONOTONIC, &(timer->start));
}


/**
* @brief Stop a sp_timer_t and update its time.
*
* @param timer The timer to stop.
*/
static inline void timer_stop(sp_timer_t * const timer)
{
  clock_gettime(CLOCK_MONOTONIC, &(timer->stop));
  timer->running = 0;
  timer->seconds += (double)(timer->stop.tv_sec - timer->start.tv_sec);
  timer->seconds += (timer->stop.tv_nsec - timer->start.tv_nsec)*1e-9;
}


/**
* @brief Give a sp_timer_t a fresh start by resetting and starting it.
*
* @param timer The timer to refresh.
*/
static inline void timer_fstart(sp_timer_t * const timer)
{
  timer_reset(timer);
  timer_start(timer);
}

#endif
