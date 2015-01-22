#ifndef SPLATT_CONVERT_H
#define SPLATT_CONVERT_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

/**
* @brief Types of tensor conversion available.
*/
typedef enum
{
  CNV_IJK_GRAPH,  /** Convert to a tri-partite graph. */
  CNV_FIB_SPMAT,  /** Convert to a CSR matrix whose rows are <mode> fibers. */
  CNV_FIB_HGRAPH, /** Convert to a hypergraph whose nodes are <mode> fibers. */
  CNV_ERROR,
} splatt_convert_type;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


/**
* @brief Load a tensor, convert to a different form, and write it to disk.
*
* @param ifname The tensor filename.
* @param ofname The output filename.
* @param mode Which mode to operate on (if applicable).
* @param type The type of conversion to perform.
*/
void tt_convert(
  char const * const ifname,
  char const * const ofname,
  idx_t const mode,
  splatt_convert_type const type);

#endif
