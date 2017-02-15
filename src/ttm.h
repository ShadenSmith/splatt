#ifndef SPLATT_TTM_H
#define SPLATT_TTM_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "csf.h"
#include "thd_info.h"


/* XXX GLOBALS */
extern idx_t ttmc_num_csf;
extern idx_t ttmc_csf_assign[MAX_NMODES];

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


#define ttmc_compute_ncols splatt_ttmc_compute_ncols
/**
* @brief Compute the number of columns in the TTMc output for all modes. The
*        total core tensor size also is written to ncols[nmodes].
*
* @param nfactors The rank of the decomposition in each mode.
* @param nmodes The number of modes.
* @param[out] ncols ncols[m] stores the number of columns in the output of the
*                   mode-m TTMc.
*/
void ttmc_compute_ncols(
    idx_t const * const nfactors,
    idx_t const nmodes,
    idx_t * const ncols);


#define make_core splatt_make_core
/**
* @brief Compute the dense core of a Tucker decomposition.
*
* @param ttmc The output of a TTMc.
* @param lastmat The last factor matrix (not included in the TTMc).
* @param[out] core The output core.
* @param nmodes The number of modes in the tensor.
* @param mode The mode of lastmat.
* @param nfactors The ranks of the factor matrices.
* @param nlongrows The length of the shared dim (#rows in lastmat).
*/
void make_core(
    val_t * ttmc,
    val_t * lastmat,
    val_t * core,
    idx_t const nmodes,
    idx_t const mode,
    idx_t const * const nfactors,
    idx_t const nlongrows);


#define permute_core splatt_permute_core
/**
* @brief Permute the dense core output from tucker_hooi_iterate. Permutation
*        is necessary because of the mode ordering done during CSF allocation.
*
* @param tensors The CSF tensor(s) used.
* @param core The core to permute.
* @param nfactors The ranks of the decomposition.
* @param opts The options used during allocation/factorization.
*/
void permute_core(
    splatt_csf const * const tensors,
    val_t * const core,
    idx_t const * const nfactors,
    double const * const opts);


#define ttmc_alloc_thds splatt_ttmc_alloc_thds
/**
* @brief Allocate thread structures for TTMc.
*
* @param nthreads The number of threads we will use.
* @param tensors The CSF tensor(s).
* @param nfactors The number of columns in each mode.
* @param opts The options used during allocation/factorization.
*
* @return Thread buffers to be freed by `thd_free()`.
*/
thd_info * ttmc_alloc_thds(
    idx_t const nthreads,
    splatt_csf const * const tensors,
    idx_t const * const nfactors,
    double const * const opts);



#define ttmc_fill_flop_tbl splatt_ttmc_fill_flop_tbl
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
    size_t table[SPLATT_MAX_NMODES][SPLATT_MAX_NMODES]);


splatt_csf * ttmc_choose_csf(
    sptensor_t * const tt,
    idx_t const * const nfactors,
    idx_t max_tensors,
    idx_t * num_chosen_tensors,
    idx_t * csf_assignments);


#endif
