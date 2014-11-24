#ifndef SPLATT_STATS_H
#define SPLATT_STATS_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"



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
void tt_stats(
  char const * const ifname,
  splatt_stats_type const type,
  idx_t const mode,
  char const * const pfile);

#endif
