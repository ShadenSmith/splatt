#ifndef SPLATT_TIMER_H
#define SPLATT_TIMER_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <time.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __MACH__
  #include <mach/mach.h>
  #include <mach/mach_time.h>
#endif


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

/**
* @brief Represents a wall-clock timer.
*/
typedef struct
{
  bool running;
  double seconds;
  double start;
  double stop;
} sp_timer_t;


/**
* @brief timer_id provides easy indexing into timers[].
*/
typedef enum
{
  TIMER_LVL0,   /* LEVEL 0 */
  TIMER_ALL,
  TIMER_CPD,
  TIMER_REORDER,
  TIMER_CONVERT,
  TIMER_LVL1,   /* LEVEL 1 */
  TIMER_MTTKRP,
  TIMER_INV,
  TIMER_FIT,
  TIMER_MATMUL,
  TIMER_ATA,
  TIMER_MATNORM,
  TIMER_IO,
  TIMER_PART,
  TIMER_LVL2,   /* LEVEL 2 */
#ifdef SPLATT_USE_MPI
  TIMER_MPI,
  TIMER_MPI_IDLE,
  TIMER_MPI_COMM,
  TIMER_MPI_ATA,
  TIMER_MPI_REDUCE,
  TIMER_MPI_PARTIALS,
  TIMER_MPI_NORM,
  TIMER_MPI_UPDATE,
  TIMER_MPI_FIT,
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


/* globals */
int timer_lvl;
sp_timer_t timers[TIMER_NTIMERS];


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define init_timers splatt_init_timers
/**
* @brief Call timer_reset() on all of timers[].
*/
void init_timers(void);


#define report_times splatt_report_times
/**
* @brief Output a summary of all used timers.
*/
void report_times(void);


#define timer_inc_verbose splatt_timer_inc_verbose
/**
* @brief Increment timer verbosity to the next level;
*/
void timer_inc_verbose(void);



/**
* @brief Return the number of seconds since an unspecified time (e.g., Unix
*        epoch). This is accomplished with a high-resolution monotonic timer,
*        suitable for performance timing.
*
* @return The number of seconds.
*/
static inline double monotonic_seconds()
{
#ifdef __MACH__
  /* OSX */
  static mach_timebase_info_data_t info;
  static double seconds_per_unit;
  if(seconds_per_unit == 0) {
    mach_timebase_info(&info);
    seconds_per_unit = (info.numer / info.denom) / 1e9;
  }
  return seconds_per_unit * mach_absolute_time();
#else
  /* Linux systems */
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}


/**
* @brief Reset all fields of a sp_timer_t.
*
* @param timer The timer to reset.
*/
static inline void timer_reset(sp_timer_t * const timer)
{
  timer->running = false;
  timer->seconds = 0;
  timer->start   = 0;
  timer->stop    = 0;
}


/**
* @brief Start a sp_timer_t. NOTE: this does not reset the timer.
*
* @param timer The timer to start.
*/
static inline void timer_start(sp_timer_t * const timer)
{
  if(!timer->running) {
    timer->running = true;
    timer->start = monotonic_seconds();
  }
}


/**
* @brief Stop a sp_timer_t and update its time.
*
* @param timer The timer to stop.
*/
static inline void timer_stop(sp_timer_t * const timer)
{
  timer->running = false;
  timer->stop = monotonic_seconds();
  timer->seconds += timer->stop - timer->start;
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
