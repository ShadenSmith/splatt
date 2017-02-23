/**
* @file api_kernels.h
* @brief Functions for performing tensor kernels (e.g., norm, MTTKRP, TTMc).
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2017-02-22
*/



#ifndef SPLATT_SPLATT_KERNELS_H
#define SPLATT_SPLATT_KERNELS_H

#include <stdbool.h>



typedef struct
{
  splatt_idx_t num_csf;
  splatt_idx_t num_threads;

  splatt_idx_t mode_csf_map[SPLATT_MAX_NMODES];

  splatt_idx_t * tile_partition[SPLATT_MAX_NMODES];
  splatt_idx_t * tree_partition[SPLATT_MAX_NMODES];

  /* Mode privatization to avoid synchronization */
  double reduction_time;
  bool is_privatized[SPLATT_MAX_NMODES];
  splatt_val_t * * privatize_buffer;
} splatt_mttkrp_ws;


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


splatt_mttkrp_ws * splatt_mttkrp_alloc_ws(
    splatt_csf const * const tensors,
    splatt_idx_t const ncolumns,
    double const * const options);


/**
* @brief Free the memory allocated for an MTTKRP workspace.
*
* @param ws The workspace to free.
*/
void splatt_mttkrp_free_ws(
    splatt_mttkrp_ws * const ws);


/** @} */


#ifdef __cplusplus
}
#endif

#endif
