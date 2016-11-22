/**
* @file api_kernels.h
* @brief Functions for performing tensor kernels (e.g., norm, MTTKRP, TTMc).
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/



#ifndef SPLATT_SPLATT_KERNELS_H
#define SPLATT_SPLATT_KERNELS_H


/*
 * KERNEL API
 */


#ifdef __cplusplus
extern "C" {
#endif


/**
\defgroup api_kernel_list List of functions for tensor kernels.
@{
*/

/**
* @brief Matricized Tensor times Khatri-Rao Product (MTTKRP) with a sparse
*        tensor in CSF format.
*
* @param mode Which mode we are operating on.
* @param ncolumns How many columns each matrix has ('nfactors').
* @param tensors The CSF tensor to multipy with.
* @param matrices The row-major dense matrices to multiply with.
* @param[out] matout The output matrix.
* @param options SPLATT options array.
*
* @return SPLATT error code. SPLATT_SUCCESS on success.
*/
int splatt_mttkrp(
    splatt_idx_t const mode,
    splatt_idx_t const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const matout,
    double const * const options);



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
* @param mode Which mode we are operating on (and skipping matrices[mode]).
* @param ncolumns Then number of columns in each matrix.
* @param tensors The CSF tensor to multiply with.
* @param matrices The row-major matrices to multipy with.
* @param tenout The output tensor.
* @param options SPLATT options array.
*
* @return SPLATT error code. SPLATT_SUCCESS on success.
*/
int splatt_ttmc(
    splatt_idx_t const mode,
    splatt_idx_t const * const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const tenout,
    double const * const options);




/**
* @brief Tensor times Matrix (TTM) chain. Multiplies every mode of a sparse
*        tensor stored in CSF by the supplied row-major matrices. The matrices
*        should have as many rows as the corresponding dimension of the tensor.
*
*        This is equivalent to the Tensor Toolbox function:
*
*        tenout = ttm(tensors, matrices, 't');
*
*        ** NOTE **
*
*        The output is a tensor has dimensions (ncolumns[0] x ...) with a
*        "row major" layout. The last mode has stride 1, the second to last
*        mode has stride ncolumns[nmodes-1], and so on. For example, if the
*        output is of size F1 x F2 x F3, entry G(i,j,k) is found at
*        tenout[(i * F2 * F3) + (j * F3) + k].
*
*
* @param ncolumns Then number of columns in each matrix.
* @param tensors The CSF tensor to multiply with.
* @param matrices The row-major matrices to multipy with.
* @param tenout The output tensor.
* @param options SPLATT options array.
*
* @return SPLATT error code. SPLATT_SUCCESS on success.
*/
int splatt_ttmc_full(
    splatt_idx_t const * const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const tenout,
    double const * const options);



/** @} */


#ifdef __cplusplus
}
#endif

#endif
