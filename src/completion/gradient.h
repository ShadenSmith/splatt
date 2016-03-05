#ifndef SPLATT_GRADIENT_H
#define SPLATT_GRADIENT_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "completion.h"




/******************************************************************************
 * GRADIENT/FIRST ORDER FUNCTIONS
 *****************************************************************************/


#define tc_gradient splatt_tc_gradient
/**
* @brief Evaluate the gradient (including regularization).
*
* @param train The CSF of training data.
* @param model The model we are evaluating.
* @param ws Workspace.
* @param[out] gradients The array of gradient values to fill.
*/
void tc_gradient(
    splatt_csf const * const train,
    tc_model const * const model,
    tc_ws * const ws,
    val_t * * gradients);


#define tc_line_search splatt_tc_line_search
/**
* @brief Perform a simple backtracking line search to find a new step size that
*        reduces the objective function.
*
* @param train The training data for evaluating objective.
* @param model The model to update.
* @param ws Workspace storing gradients.
* @param prev_obj The previous objective value.
* @param gradients The gradients of the current solution.
* @param directions The direction of the step to take. This can point to
*                   'gradients' for a steepest descent step.
* @param[out] ret_loss On exit, stores the new loss value.
* @param[out] ret_frobsq On exit, stores the new \sum \lambda||A||_F^2 penalty.
*/
void tc_line_search(
    sptensor_t const * const train,
    tc_model * const model,
    tc_ws * const ws,
    val_t const prev_obj,
    val_t * * gradients,
    val_t * * directions,
    val_t * ret_loss,
    val_t * ret_frobsq);

#endif
