#ifndef SPLATT_TIMER_H
#define SPLATT_TIMER_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <sys/time.h>
#include <stddef.h>


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
  struct timeval start;
  struct timeval stop;
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


extern int timer_lvl;
extern sp_timer_t timers[];


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


#define reset_cpd_timers splatt_reset_cpd_timers
/**
* @brief Resets serial and MPI timers that were activated during some CPD
*        pre-processing.
*/
void reset_cpd_timers();


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
  timer->start.tv_usec = 0;
  timer->stop.tv_sec   = 0;
  timer->stop.tv_usec  = 0;
}


/**
* @brief Start a sp_timer_t. NOTE: this does not reset the timer.
*
* @param timer The timer to start.
*/
static inline void timer_start(sp_timer_t * const timer)
{
  timer->running = 1;
  gettimeofday(&(timer->start), NULL);
}


/**
* @brief Stop a sp_timer_t and update its time.
*
* @param timer The timer to stop.
*/
static inline void timer_stop(sp_timer_t * const timer)
{
  timer->running = 0;
  gettimeofday(&(timer->stop), NULL);
  timer->seconds += (double)(timer->stop.tv_sec - timer->start.tv_sec);
  timer->seconds += 1e-6 * (timer->stop.tv_usec - timer->start.tv_usec);
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
