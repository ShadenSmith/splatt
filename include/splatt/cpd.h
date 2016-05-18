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
  SPLATT_REG_NONE,
  SPLATT_REG_L1,
  SPLATT_REG_L2,
  SPLATT_REG_NONNEG,

#if 0
  SPLATT_REG_SMOOTHNESS,
  SPLATT_REG_SYMMETRY,
  SPLATT_REG_SIMPLEX,
  SPLATT_REG_CUSTOM,
#endif

} splatt_reg_type;



typedef struct
{
  splatt_reg_type which;

  /* Arbitrary data -- often a lambda parameter. */
  void * data;
} splatt_regularization;


typedef struct
{
  /* convergence */
  splatt_val_t tolerance;
  splatt_idx_t max_iterations;

  /* inner ADMM solves */
  splatt_val_t inner_tolerance;
  splatt_idx_t max_inner_iterations;

  /* constraints */
  splatt_regularization constraints[SPLATT_MAX_NMODES];

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
