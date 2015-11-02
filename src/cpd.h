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
#define cpd_als_iterate splatt_cpd_als_iterate
double cpd_als_iterate(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  val_t * const lambda,
  idx_t const nfactors,
  rank_info * const rinfo,
  double const * const opts);

#define cpd_post_process splatt_cpd_post_process
void cpd_post_process(
  idx_t const nfactors,
  idx_t const nmodes,
  matrix_t ** mats,
  val_t * const lambda,
  thd_info * const thds,
  idx_t const nthreads,
  rank_info * const rinfo);

#endif
