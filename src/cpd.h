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
* @param[out] mats The output factors.
* @param[out] lambda The output vector for scaling.
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


#define cpd_post_process splatt_cpd_post_process
/**
* @brief Perform a final normalization of the factor matrices and gather into
*        lambda.
*
*        TODO: Sort factor columns and lambda.
*
* @param nfactors The rank of the decomposition.
* @param nmodes The number of modes of the tensor.
* @param[out] mats The output factors.
* @param[out] lambda The output vector for scaling.
* @param thds Thread buffers.
* @param nthreads The number of threads to use.
* @param rinfo MPI rank information (not used, TODO remove).
*/
void cpd_post_process(
  idx_t const nfactors,
  idx_t const nmodes,
  matrix_t ** mats,
  val_t * const lambda,
  thd_info * const thds,
  idx_t const nthreads,
  rank_info * const rinfo);

#endif
