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
typedef struct
{
  void * scratch;
  sp_timer_t ttime;
} thd_info;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void thd_times(
  thd_info * thds,
  idx_t const nthreads);

void thd_reset(
  thd_info * thds,
  idx_t const nthreads);

thd_info * thd_init(
  idx_t const nthreads,
  idx_t const scratch_bytes);

void thd_free(
  thd_info * thds,
  idx_t const nthreads);

#endif
