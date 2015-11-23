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

#define ttmc_csf splatt_ttmc_csf
/**
* @brief Tensor times Matrix (TTM) chain. Multiplies a sparse tensor in CSF
*        form by all matrices except one.
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
void ttmc_csf(
    splatt_csf const * const tensors,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts);


#define ttmc_stream splatt_ttmc_stream
/**
* @brief Tensor times Matrix (TTM) chain. Multiplies a sparse tensor in
*        coordinate form by all matrices except one.
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
* @param tensors The coordinate tensor to use.
* @param mats The input matrices.
* @param tenout The output tensor.
* @param mode Which mode we are computing for.
* @param opts SPLATT options.
*/
void ttmc_stream(
    sptensor_t const * const tt,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    double const * const opts);


#define tenout_dim splatt_tenout_dim
/**
* @brief Compute the maximum needed size of tenout. The max size is based on
*        performing TTM on any mode.
*
* @param nmodes The number of modes.
* @param nfactors The rank (# columns) of each factor matrix.
* @param dims The dimensions of the tensor.
*
* @return The number of elements to allocate for ttm().
*/
idx_t tenout_dim(
    idx_t const nmodes,
    idx_t const * const nfactors,
    idx_t const * const dims);


#endif
