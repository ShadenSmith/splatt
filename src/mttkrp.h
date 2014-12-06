#ifndef SPLATT_MTTKRP_H
#define SPLATT_MTTKRP_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "ftensor.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void splatt_mttkrp(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode);


#endif
