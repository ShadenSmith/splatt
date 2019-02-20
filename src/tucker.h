#ifndef SPLATT_TUCKER_H
#define SPLATT_TUCKER_H

#include "base.h"


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "matrix.h"
#include "splatt_mpi.h"

typedef enum
{
  TUCKER_CSF_ALLOC_SIMPLE,
  TUCKER_CSF_ALLOC_GREEDY,
  TUCKER_CSF_ALLOC_ITER,
} tucker_csf_alloc_t;


/* XXX GLOBALS */
extern idx_t ttmc_max_csf;
extern tucker_csf_alloc_t tucker_alloc_policy;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define tucker_hooi_iterate splatt_tucker_hooi_iterate
/**
* @brief The primary computation in TUCKER-ALS. API functions call this one.
*
* @param tensors The COO tensor to factor.
* @param[out] mats The output factors.
* @param[out] core The output core tensor.
* @param nfactors The rank of the factorization.
* @param opts SPLATT options array.
*
* @return The final fitness of the factorization.
*/
double tucker_hooi_iterate(
    splatt_coo * const coo,
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

#endif
