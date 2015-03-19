#ifndef SPLATT_STATS_H
#define SPLATT_STATS_H

#include "base.h"



/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

/**
* @brief The types of tensor statistics available.
*/
typedef enum
{
  STATS_BASIC,    /** Dimensions, nonzero count, and density. */
  STATS_HPARTS,   /** Hypergraph partitioning information. Requires MODE */
  STATS_ERROR,
} splatt_stats_type;



/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Output statistics about a sparse tensor.
*
* @param tt The sparse tensor to inspect.
* @param ifname The filename of the tensor. Can be NULL.
* @param type The type of statistics to output.
* @param mode The mode of tt to operate on, if applicable.
* @param pfile The partitioning file to work with, if applicable.
*/
void stats_tt(
  sptensor_t * const tt,
  char const * const ifname,
  splatt_stats_type const type,
  idx_t const mode,
  char const * const pfile);

#endif
