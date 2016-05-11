/**
* @file api_factorization.h
* @brief Functinos for performing tensor factorizations.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/



#ifndef SPLATT_SPLATT_FACTORIZATION_H
#define SPLATT_SPLATT_FACTORIZATION_H


/*
 * FACTORIZATION API
 */


#ifdef __cplusplus
extern "C" {
#endif


/**
\defgroup api_fact_list List of functions for tensor factorizations.
@{
*/


/**
* @brief Compute the CPD using alternating least squares.
*
* @param tensors An array of splatt_csf created by SPLATT.
* @param nfactors The rank of the decomposition to perform.
* @param options Options array for SPLATT.
* @param[out] factored The factored tensor in Kruskal format.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_cpd_als(
    splatt_csf const * const tensors,
    splatt_idx_t const nfactors,
    double const * const options,
    splatt_kruskal * factored);


/**
* @brief Compute the Tucker decomposition using alternating least squares.
*
* @param nfactors The number of factors to use for each mode.
* @param nmodes The number of modes in the tensor.
* @param tensors An array of splatt_csf created by SPLATT.
* @param options Options array for SPLATT.
* @param factored The factored tensor in Kruskal format.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_tucker_als(
    splatt_idx_t const * const nfactors,
    splatt_idx_t const nmodes,
    splatt_csf const * const tensors,
    double const * const options,
    splatt_tucker_t * factored);



/**
* @brief Free a splatt_tucker_t allocated by splatt_tucker().
*
* @param factored The factored tensor to free.
*/
void splatt_free_tucker(
    splatt_tucker_t * factored);


/** @} */


#ifdef __cplusplus
}
#endif

#endif
