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
  SPLATT_CON_NONE,
  SPLATT_REG_L1,
  SPLATT_REG_L2,
  SPLATT_CON_NONNEG,
  SPLATT_REG_SMOOTHNESS,

#if 0
  SPLATT_CON_SYMMETRY,
  SPLATT_CON_SIMPLEX,
  SPLATT_CON_CUSTOM,
#endif

} splatt_con_type;



typedef struct
{
  splatt_con_type which;

  /* Arbitrary data -- often a lambda parameter. */
  void * data;
} splatt_constraint;


typedef struct
{
  /* convergence */
  splatt_val_t tolerance;
  splatt_idx_t max_iterations;

  /* inner ADMM solves */
  splatt_val_t inner_tolerance;
  splatt_idx_t max_inner_iterations;

  /* constraints */
  bool unconstrained; /* true if NO constraints or regularizations applied */
  splatt_constraint constraints[SPLATT_MAX_NMODES];
  
  /* chunked AO-ADMM. 0=unchunked */
  splatt_idx_t chunk_sizes[SPLATT_MAX_NMODES];
} splatt_cpd_opts;




/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif



/**
 * \defgroup CPD_opts Functions for configuring a CPD factorization.
 *  @{
 */

/**
* @brief Allocate a `splatt_cpd_opts` structure and initialize with sane
*        defaults.
*
* @return The allocated options, to be freed by `splatt_free_cpd_opts()`.
*/
splatt_cpd_opts * splatt_alloc_cpd_opts(void);

/**
* @brief Free the memory allocated by `splatt_alloc_cpd_opts()`.
*
* @param opts The structure to free.
*/
void splatt_free_cpd_opts(
    splatt_cpd_opts * const opts);



void splatt_cpd_reg_l1(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale);


void splatt_cpd_reg_l2(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale);


void splatt_cpd_reg_smooth(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale);



void splatt_cpd_con_nonneg(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode);


void splatt_cpd_con_clear(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode);

/** }@ */



/**
* @brief Allocate the memory required for a CPD factorization of a given tensor.
*
* @param tensor The tensor to factor.
* @param rank The desired factorization rank.
*
* @return Allocated factors.
*/
splatt_kruskal * splatt_alloc_cpd(
    splatt_csf const * const tensor,
    splatt_idx_t rank);


/**
* @brief Free a splatt_kruskal allocated by `splatt_alloc_cpd()`.
*
* @param factored The factored tensor to free.
*/
void splatt_free_cpd(
    splatt_kruskal * factored);


/**
* @brief Compute a CPD factorization.
*
* @param tensor The tensor to factor.
* @param rank The desired factorization rank.
* @param cpd_opts Configuration for the specific factorization. Can be NULL for
*                 default options.
* @param global_opts Configuration for general SPLATT behavior. Can be NULL for
*                    default options.
* @param[out] factored The factored tensor.
*
* @return SPLATT error code. SPLATT_SUCCESS if no error.
*
* @ingroup splatt_factorizations
*/
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
