#ifndef SPLATT_CONVERT_H
#define SPLATT_CONVERT_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef enum
{
  CNV_IJK_GRAPH,
  CNV_FIB_HGRAPH,
  CNV_ERROR,
} splatt_convert_type;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_convert(
  char const * const ifname,
  char const * const ofname,
  idx_t const mode,
  splatt_convert_type const type);

#endif
