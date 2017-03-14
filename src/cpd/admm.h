/**
* @file admm.h
* @brief Functions for ADMM inner iterations during AO-ADMM.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-16
*/


#ifndef SPLATT_CPD_ADMM_H
#define SPLATT_CPD_ADMM_H



/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "cpd.h"

#include "../matrix.h"
#include "../thd_info.h"


static bool ADMM_ROW_CONVERGE = false;




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

val_t admm_inner(
    idx_t mode,
    matrix_t * * mats,
    val_t * const restrict column_weights,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts);



#define cpd_init_constraints splatt_cpd_init_constraints
/**
* @brief Call init_func() on all constraints.
*
* @param opts CPD parameters containing the constraints.
* @param primals Primal matrices (the factors).
* @param nmodes The number of modes.
*/
void cpd_init_constraints(
    splatt_cpd_opts * const opts,
    matrix_t * * primals,
    idx_t const nmodes);


#define cpd_finalize_constraints splatt_cpd_finalize_constraints
/**
* @brief Call post_func() on all constraints.
*
* @param opts CPD parameters containing the constraints.
* @param primals Primal matrices (the factors).
* @param nmodes The number of modes.
*/
void cpd_finalize_constraints(
    splatt_cpd_opts * const opts,
    matrix_t * * primals,
    idx_t const nmodes);

#endif
