/**
* @file cpd.h
* @brief API functions for computing CPD factorizations.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-15
*/

#ifndef SPLATT_CPD_CPD_H
#define SPLATT_CPD_CPD_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../matrix.h"
#include "../thd_info.h"




/******************************************************************************
 * TYPES
 *****************************************************************************/


/**
* @brief A workspace used for computing CPD factorizations.
*/
typedef struct
{
  idx_t nmodes; /** The number of modes in the factorization. */
  matrix_t * aTa[MAX_NMODES]; /** Caching for the A^T * A factors. */
  matrix_t * aTa_buf;         /** Buffer space for accumulating hada(aTa[:]).*/
  matrix_t * gram;            /** The gram matrix to invert. */

  matrix_t * mttkrp_buf;      /** The output of the MTTKRP operation. */

  int nthreads; /** The number of threads we are using. */
  thd_info * thds; /** Thread private structures. */

  /* AO-ADMM */

  matrix_t * auxil; /** Auxiliary matrix for AO-ADMM factorization. */
  matrix_t * duals[MAX_NMODES]; /** Dual matrices for AO-ADMM factorization. */
  matrix_t * mat_init; /** Store the initial primal variable each ADMM iteration. */
} cpd_ws;




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


#define cpd_norm splatt_cpd_norm
/**
* @brief Compute the squared Frobenius norm of a CPD factorization.
*
* @param ws CPD workspace -- uses aTa[:] and aTa_buf.
* @param column_weights The weights of the CPD rank-one factors.
*
* @return The squared Frobenius norm.
*/
val_t cpd_norm(
    cpd_ws const * const ws,
    val_t const * const restrict column_weights);



#define cpd_innerprod splatt_cpd_innerprod
/**
* @brief Compute the inner product of a CPD factorization and a CSF tensor.
*        This uses a cached MTTKRP to save many flops.
*
* @param lastmode The last mode that the MTTKRP was performed on.
* @param ws CPD workspace -- ws->mttkrp_buf is used.
* @param mats The CPD factors.
* @param column_weights The weights of the rank-one factors.
*
* @return The inner product.
*/
val_t cpd_innerprod(
    idx_t lastmode,
    cpd_ws const * const ws,
    matrix_t * * mats,
    val_t const * const restrict column_weights);



#define cpd_iterate splatt_cpd_iterate
/**
* @brief The primary computation in CPD AO-ADMM. API functions call this one.
*
* @param tensor The CSF tensor to factor.
* @param rank The rank of the factorization.
* @param ws CPD workspace.
* @param cpd_opts CPD factorization parameters.
* @param global_opts SPLATT global parameters.
* @param[out] factored The factored tensor.
*
* @return The fitness of the factorization: 1 - relative error.
*/
double cpd_iterate(
    splatt_csf const * const tensor,
    idx_t rank,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts,
    splatt_kruskal * factored);



#define cpd_post_process splatt_cpd_post_process
/**
* @brief Perform a final normalization of the factor matrices and gather into
*        column_weights.
*
*        TODO: Sort columns weights and factor columns.
*
* @param[out] mats The factor matrices.
* @param[out] column_weights The weights of the rank-one factors.
* @param ws CPD workspace.
* @param cpd_opts CPD options.
* @param global_opts SPLATT options.
*/
void cpd_post_process(
    matrix_t * * mats,
    val_t * const column_weights,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts);




/**
* @brief Allocate a workspace for computing a CPD factorization. Tensor, CPD,
*        and global option info is collected to ease future extension.
*
* @param tensor The tensor we are factoring.
* @param rank The rank of the CPD factorization.
* @param cpd_opts CPD options (constraints will affect workspace).
* @param global_opts Global configuration options (# threads, etc. used)
*
* @return The allocated CPD workspace, to be freed by `cpd_free_ws()`.
*/
cpd_ws * cpd_alloc_ws(
    splatt_csf const * const tensor,
    idx_t rank,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts);


/**
* @brief Free the workspace allocated by `cpd_alloc_ws()`.
*
* @param ws The workspace to free.
*/
void cpd_free_ws(
    cpd_ws * const ws);



#endif

