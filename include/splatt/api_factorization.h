/**
* @file api_factorization.h
* @brief Functinos for performing tensor factorizations.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/



#ifndef SPLATT_SPLATT_FACTORIZATION_H
#define SPLATT_SPLATT_FACTORIZATION_H

typedef enum
{
  SPLATT_REG_FROBENIUS,
  SPLATT_REG_LASSO,
  SPLATT_REG_SMOOTH,

  SPLATT_REG_NUMREGS
} splatt_regularize_type;


typedef enum
{
  SPLATT_CON_NONNEG,
  SPLATT_CON_SYMMETRY,
  SPLATT_CON_SIMPLEX,

  SPLATT_CON_NUMCONS
} splatt_constraint_type;


typedef struct
{
  splatt_idx_t rank;

  /* convergence */
  splatt_val_t tolerance;
  splatt_idx_t max_iterations;

  splatt_verbosity_type verbosity;

} splatt_cpd_opts;


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



splatt_error_type splatt_cpd(
    splatt_csf const * const tensor,
    splatt_cpd_opts const * const opts,
    splatt_kruskal * factored);




/** @} */


#ifdef __cplusplus
}
#endif

#endif
