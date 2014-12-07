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

void mttkrp_splatt(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode);

void mttkrp_ttbox(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch);

#endif
