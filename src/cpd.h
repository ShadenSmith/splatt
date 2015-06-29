#ifndef SPLATT_CPD_H
#define SPLATT_CPD_H

#include "base.h"


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "ftensor.h"
#include "matrix.h"
#include "splatt_mpi.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
#define cpd_als splatt_cpd_als
val_t cpd_als(
  ftensor_t ** ft,
  matrix_t ** mats,
  matrix_t ** globmats,
  val_t * const lambda,
  idx_t const nfactors,
  rank_info * const rinfo,
  double const * const opts);



#endif
