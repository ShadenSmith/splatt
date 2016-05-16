/**
* @file cpd.h
* @brief Functions for computing CPD factorizations.
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

typedef struct
{
  idx_t nmodes;
  matrix_t * aTa[MAX_NMODES];
  matrix_t * aTa_buf;

  matrix_t * mttkrp_buf;

  int nthreads;
  thd_info * thds;

  /* AO-ADMM */
  matrix_t * auxil[MAX_NMODES];
  matrix_t * duals[MAX_NMODES];
} cpd_ws;




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

val_t cpd_norm(
    cpd_ws const * const ws,
    val_t const * const restrict column_weights);


double cpd_iterate(
    splatt_csf const * const tensor,
    idx_t rank,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts,
    splatt_kruskal * factored);


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

