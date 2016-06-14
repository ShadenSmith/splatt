#ifndef SPLATT_TUCKER_H
#define SPLATT_TUCKER_H

#include "base.h"


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "matrix.h"
#include "splatt_mpi.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define tucker_hooi_iterate splatt_tucker_hooi_iterate
/**
* @brief The primary computation in TUCKER-ALS. API functions call this one.
*
* @param tensors The CSF tensor(s) to factor.
* @param[out] mats The output factors.
* @param[out] core The output core tensor.
* @param nfactors The rank of the factorization.
* @param opts SPLATT options array.
*
* @return The final fitness of the factorization.
*/
double tucker_hooi_iterate(
    splatt_csf const * const tensors,
    matrix_t ** mats,
    val_t * const core,
    idx_t const * const nfactors,
    double const * const opts);


#define tucker_alloc_thds splatt_tucker_alloc_thds
/**
* @brief Allocate threads structures used in the Tucker decomposition.
*
* @param nthreads The number of threads to allocate.
* @param tensors The CSF tensor(s).
* @param nfactors The ranks of the decomposition.
* @param opts The options used during CSF allocation.
*
* @return Thread structures.
*/
thd_info * tucker_alloc_thds(
    idx_t const nthreads,
    splatt_csf const * const tensors,
    idx_t const * const nfactors,
    double const * const opts);


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



#endif