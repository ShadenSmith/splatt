/**
* @file api_factorization.h
* @brief Functions for performing tensor factorizations.
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


/** @} */


#ifdef __cplusplus
}
#endif

#endif
