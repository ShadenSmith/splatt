#ifndef SPLATT_COMPLETE_H
#define SPLATT_COMPLETE_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "base.h"
#include "matrix.h"
#include "sptensor.h"




/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

typedef struct
{
  idx_t rank;
  idx_t nmodes;
  idx_t dims[MAX_NMODES];
  val_t * factors;
} tc_model;



typedef struct
{
  val_t learn_rate;
  idx_t max_its;
  val_t regularization[MAX_NMODES];

  thd_info * thds;
} tc_ws;




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


#define tc_rmse splatt_tc_rmse
/**
* @brief Compute the RMSE against of a factorization against a test tensor.
*        RMSE is defined as: sqrt( tc_loss_sq() / nnz ).
*
* @param test The tensor to test against.
* @param model The factorization the evaluate.
* @param ws Workspace to use (thread buffers are accessed).
*
* @return The RMSE.
*/
val_t tc_rmse(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws);



#define tc_loss_sq splatt_tc_loss_sq
/**
* @brief Compute the sum-of-squared loss of a factorization. Computes:
*           sum((observed-predicted)^2) over all nonzeros in 'test'.
*
* @param test The tensor to test against.
* @param model The factorization the evaluate.
* @param ws Workspace to use (thread buffers are accessed).
*
* @return The loss.
*/
val_t tc_loss_sq(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws);



#define tc_frob_sq splatt_tc_frob_sq
/**
* @brief Compute the sum of the squared Frobenius norms of a model, including
*        regularization penalties.
*
* @param model The factorization to evaluate.
* @param ws Workspace to use (regularization[:] are accessed).
*
* @return \sum_{m=1}^{M} \lambda_m || A^{(m)} ||_F_2.
*/
val_t tc_frob_sq(
    tc_model const * const model,
    tc_ws const * const ws);



#define predict_val splatt_predict_val
/**
* @brief Predict a value of the nonzero in position 'index'.
*
* @param model The model to use for prediction.
* @param tt The sparse tensor to test against.
* @param index The index of the nonzero to predict. tt->ind[:][index] are used.
* @param buffer A buffer at least of size model->rank.
*
* @return The predicted value.
*/
val_t tc_predict_val(
    tc_model const * const model,
    sptensor_t const * const tt,
    idx_t const index,
    val_t * const restrict buffer);


#endif
