#ifndef SPLATT_TTM_H
#define SPLATT_TTM_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "csf.h"
#include "thd_info.h"




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define ttm_csf splatt_ttm_csf
/**
* @brief Tensor times Matrix (TTM). Multiplices a sparse tensor in CSF form
*        by all matrices except one.
*
*        ** NOTE **
*
*        The output is a tensor has dimensions (dims[mode] x ncolumns[0] x ...)
*        The original dimension is placed first, contrary to the behavior of
*        Tensor Toolbox and other software packages. This makes the data layout
*        'automatically' the unfolded tensor for the proceeding SVD calculation
*        during Tucker factorization.
*
*
* @param tensors The CSF tensor(s) to factor.
* @param mats The input matrices.
* @param tenout The output tensor.
* @param mode Which mode we are computing for.
* @param thds Thread structures.
* @param opts SPLATT options. This uses SPLATT_OPTION_CSF_ALLOC.
*/
void ttm_csf(
    splatt_csf const * const tensors,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts);

#endif
