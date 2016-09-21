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




#endif
