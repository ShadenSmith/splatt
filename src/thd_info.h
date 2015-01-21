#ifndef SPLATT_THDINFO_H
#define SPLATT_THDINFO_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "timer.h"



/******************************************************************************
 * PUBLIC STRUCTURES
 *****************************************************************************/

/**
* @brief A general structure for data structures that need to be thread-local.
*/
typedef struct
{
  void * scratch;
  void * scratch2;
  sp_timer_t ttime;
} thd_info;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

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
* @param scratch1_bytes The number of bytes per thread to allocate for scratch1.
* @param scratch2_bytes The number of bytes per thread to allocate for scratch2.
*
* @return A pointer to an array of thd_info.
*/
thd_info * thd_init(
  idx_t const nthreads,
  idx_t const scratch1_bytes,
  idx_t const scratch2_bytes);


/**
* @brief Allocate and initialize a number thd_info structs.
*
* @param nthreads The number of threads to allocate for.
* @param scratch1_bytes The number of bytes per thread to allocate for scratch1.
* @param scratch2_bytes The number of bytes per thread to allocate for scratch2.
*
* @return A pointer to an array of thd_info.
*/
void thd_free(
  thd_info * thds,
  idx_t const nthreads);

#endif
