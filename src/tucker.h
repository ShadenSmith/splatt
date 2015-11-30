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


#define permute_core splatt_permute_core
void permute_core(
    splatt_csf const * const tensors,
    val_t * const core,
    idx_t const * const nfactors,
    double const * const opts);


#endif
