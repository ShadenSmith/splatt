#ifndef SPLATT_THDINFO_H
#define SPLATT_THDINFO_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "timer.h"

#include <stdarg.h>


/******************************************************************************
 * PUBLIC STRUCTURES
 *****************************************************************************/

/**
* @brief A general structure for data structures that need to be thread-local.
*/
typedef struct
{
  idx_t nscratch;
  void ** scratch;
  sp_timer_t ttime;
} thd_info;


/**
* @brief The types of parallel reductions supported.
*/
typedef enum
{
  REDUCE_SUM,
  REDUCE_MAX,
  REDUCE_ERR
} splatt_reduce_type;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Perform a parallel reduction on thds->scratch[scratchid].
*
* @param thds The thread structure we are using in the reduction.
* @param scratchid Which scratch array to reduce.
* @param nelems How many elements in the scratch array.
* @param which Which reduction operation to perform.
*/
void thd_reduce(
  thd_info * const thds,
  idx_t const scratchid,
  idx_t const nelems,
  splatt_reduce_type const which);

/**
* @brief Output a summary to STDOUT of all thread timers.
*
* @param thds The array on thd_info structs.
* @param nthreads The number of timers to print.
*/
void thd_times(
  thd_info * thds,
  idx_t const nthreads);


/**
* @brief Reset an array of thread timers.
*
* @param thds The array of thd_info structs.
* @param nthreads The number of times to reset.
*/
void thd_reset(
  thd_info * thds,
  idx_t const nthreads);


/**
* @brief Allocate and initialize a number thd_info structs.
*
* @param nthreads The number of threads to allocate for.
* @param nscratch The number of scratch arrays to use.
* @param ...      The number of bytes to allocate for each scratch array.
*
* @return A pointer to an array of thd_info.
*/
thd_info * thd_init(
  idx_t const nthreads,
  idx_t const nscratch,
  ...);

void thd_free(
  thd_info * thds,
  idx_t const nthreads);

#endif
