#ifndef SPLATT_KRUSKAL_H
#define SPLATT_KRUSKAL_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "base.h"
#include "matrix.h"
#include "sptensor.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


#define kruskal_norm splatt_kruskal_norm
/**
* @brief Find the Frobenius norm squared of a Kruskal tensor. This equivalent
*        to via computing <X,X>, the inner product of X with itself. We find
*        this via \lambda^T (AtA * BtB * ...) \lambda, where * is the Hadamard
*        product.
*
* @param nmodes The number of modes in the tensor.
* @param lambda The vector of column norms.
* @param aTa An array of Gram Matrices (AtA, BtB, ...).
*
* @return The Frobenius norm of X, squared.
*/
val_t kruskal_norm(
  idx_t const nmodes,
  val_t const * const restrict lambda,
  matrix_t ** aTa);


#define kruskal_calc_fit splatt_kruskal_calc_fit
/**
* @brief Compute the fit of a Kruskal tensor, Z, to an input tensor, X. This
*        is computed via 1 - [sqrt(<X,X> + <Z,Z> - 2<X,Z>) / sqrt(<X,X>)].
*
* @param nmodes The number of modes in the input tensors.
* @param rinfo MPI rank information.
* @param thds OpenMP thread data structures.
* @param ttnormsq The norm (squared) of the original input tensor, <X,X>.
* @param lambda The vector of column norms.
* @param mats The Kruskal-tensor matrices.
* @param mttkrp The result of doing MTTKRP along the last mode.
* @param aTa An array of matrices (length MAX_NMODES) containing BtB, CtC, etc.
*
* @return The inner product of the two tensors, computed via:
*         \lambda^T hadamard(mats[nmodes-1], mttkrp) \lambda.
*/
val_t kruskal_calc_fit(
  idx_t const nmodes,
  rank_info * const rinfo,
  thd_info * const thds,
  val_t const ttnormsq,
  val_t const * const restrict lambda,
  matrix_t ** mats,
  matrix_t const * const mttkrp,
  matrix_t ** aTa);



#define kruskal_mttkrp_inner splatt_kruskal_mttkrp_inner
/**
* @brief Compute the inner product of a Kruskal tensor and an unfactored
*        tensor. Assumes that 'm1' contains the MTTKRP result along the last
*        mode of the two input tensors. This naturally follows the end of a
*        CPD iteration.
*
* @param nmodes The number of modes in the input tensors.
* @param rinfo MPI rank information.
* @param thds OpenMP thread data structures.
* @param lambda The vector of column norms.
* @param mats The Kruskal-tensor matrices.
* @param m1 The result of doing MTTKRP along the last mode.
*
* @return The inner product of the two tensors, computed via:
*         1^T hadamard(mats[nmodes-1], m1) \lambda.
*/
val_t kruskal_mttkrp_inner(
  idx_t const nmodes,
  rank_info * const rinfo,
  thd_info * const thds,
  val_t const * const restrict lambda,
  matrix_t ** mats,
  matrix_t const * const m1);

#endif
