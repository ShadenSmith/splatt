/**
* @file api_cpd.h
* @brief Functions and structures related to the CPD factorization.
* @author Shaden Smith
* @version 2.0.0
* @date 2016-05-14
*/


#ifndef SPLATT_API_CPD_H
#define SPLATT_API_CPD_H




/******************************************************************************
 * OPTIONS
 *****************************************************************************/


typedef enum
{
  SPLATT_REG_FROBENIUS,
#if 0
  SPLATT_REG_LASSO,
  SPLATT_REG_SMOOTH,
#endif

  SPLATT_REG_NUMREGS
} splatt_regularize_type;


typedef enum
{
  SPLATT_CON_NONNEG,
#if 0
  SPLATT_CON_SYMMETRY,
  SPLATT_CON_SIMPLEX,
#endif

  SPLATT_CON_NUMCONS
} splatt_constraint_type;


typedef struct
{
  /* convergence */
  splatt_val_t tolerance;
  splatt_idx_t max_iterations;
  /* inner ADMM solves */
  splatt_val_t inner_tolerance;
  splatt_idx_t max_inner_iterations;

  /* constraints */

} splatt_cpd_opts;





/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif


/**
* @brief Allocate a `splatt_cpd_opts` structure and initialize with sane
*        defaults.
*
* @return The allocated options, to be freed by `splatt_free_cpd_opts()`.
*/
splatt_cpd_opts * splatt_alloc_cpd_opts(void);


splatt_kruskal * splatt_alloc_cpd(
    splatt_csf const * const csf,
    splatt_idx_t rank);

/**
* @brief Free the memory allocated by `splatt_alloc_cpd_opts()`.
*
* @param opts The structure to free.
*/
void splatt_free_cpd_opts(
    splatt_cpd_opts * const opts);



splatt_error_type splatt_cpd(
    splatt_csf const * const tensor,
    splatt_idx_t rank,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts,
    splatt_kruskal * factored);




#ifdef __cplusplus
}
#endif


#endif
