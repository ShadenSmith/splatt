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
/**
* @brief The primary computation in CPD-ALS. API functions call this one.
*
* @param tensors The CSF tensor(s) to factor.
* @param mats [OUT] The output factors.
* @param lambda [OUT] The output vector for scaling.
* @param nfactors The rank of the factorization.
* @param rinfo MPI rank information (not used, TODO remove).
* @param opts SPLATT options array.
*
* @return The final fitness of the factorization.
*/
double cpd_als_iterate(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  val_t * const lambda,
  idx_t const nfactors,
  rank_info * const rinfo,
  double const * const opts);


#endif
