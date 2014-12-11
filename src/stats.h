#ifndef SPLATT_STATS_H
#define SPLATT_STATS_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"



/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef enum
{
  STATS_BASIC,
  STATS_FIBERS,
  STATS_HPARTS,
  STATS_ERROR
} splatt_stats_type;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void stats_tt(
  sptensor_t * const tt,
  char const * const ifname,
  splatt_stats_type const type,
  idx_t const mode,
  char const * const pfile);

#endif
