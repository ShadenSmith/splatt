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


#define ttmc_largest_outer splatt_ttmc_largest_outer
/**
* @brief Fill 'outer_sizes' with the largest outer product (# vectors) that
*        is needed during TTMc, for each mode. This is equivalent to the
*        largest number of children that any node has.
*
* @param tensors The CSF tensor(s) to analyze.
* @param[out] outer_sizes An array to fill with the maximum # of outer products
*             per mode.
* @param opts SPLATT options.
*/
void ttmc_largest_outer(
    splatt_csf const * const tensors,
    idx_t * const outer_sizes,
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


/**
* @brief Fill at able with flop counts for TTMc (assuming no caching). The
*        table must have SPLATT_MAX_NMODES^2 rows and columns. table[i][j]
*        stores the number of flops required to compute splatt_ttmc for the
*        j'th mode, if the mode-i CSF is stored.
*
* @param tt The sparse tensor we are factoring.
* @param nfactors The rank to use for each mode.
* @param table The nmodes x nmodes table to fill. Should be declared
*               idx_t table[SPLATT_MAX_NMODES][SPLATT_MAX_NMODES].
*/
void ttmc_fill_flop_tbl(
    sptensor_t * const tt,
    idx_t const * const nfactors,
    idx_t table[SPLATT_MAX_NMODES][SPLATT_MAX_NMODES]);



#define ttmc_csf_count_flops splatt_ttmc_csf_count_flops
/**
* @brief Count the number of floating point operations required to perform
*        a TTMc operation with a CSF tensor on given mode.
*
* @param csf The CSF tensor we are using.
* @param mode The mode of interest.
* @param nfactors The ranks of the factorization.
*
* @return The number of floating point operations.
*/
idx_t ttmc_csf_count_flops(
    splatt_csf const * const csf,
    idx_t const mode,
    idx_t const * const nfactors);


#define ttmc_coord_count_flops splatt_ttmc_coord_count_flops
/**
* @brief Count the number of floating point operations required to perform
*        a TTMc operation with a coordinate tensor on given mode.
*
* @param tt The coordinate tensor we are using.
* @param mode The mode of interest.
* @param nfactors The ranks of the factorization.
*
* @return The number of floating point operations.
*/
idx_t ttmc_coord_count_flops(
    sptensor_t const * const tt,
    idx_t const mode,
    idx_t const * const nfactors);


#endif
