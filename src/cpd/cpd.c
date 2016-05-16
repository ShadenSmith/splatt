/**
* @file cpd.c
* @brief Tensor factorization with the CPD model using AO-ADMM.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-14
*/




/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "cpd.h"
#include "admm.h"

#include "../csf.h"
#include "../mttkrp.h"
#include "../timer.h"
#include "../util.h"




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/






/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


splatt_error_type splatt_cpd(
    splatt_csf const * const tensor,
    splatt_idx_t rank,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts,
    splatt_kruskal * factored)
{
  splatt_omp_set_num_threads(global_opts->num_threads);
  cpd_ws * ws = cpd_alloc_ws(tensor, rank, cpd_opts, global_opts);

  cpd_iterate(tensor, rank, ws, cpd_opts, global_opts, factored);

  /* clean up workspace */
  cpd_free_ws(ws);

  return SPLATT_SUCCESS;
}



splatt_cpd_opts * splatt_alloc_cpd_opts(void)
{
  splatt_cpd_opts * opts = splatt_malloc(sizeof(*opts));

  /* defaults */
  opts->tolerance = 1e-5;
  opts->max_iterations = 5;

  return opts;
}


void splatt_free_cpd_opts(
    splatt_cpd_opts * opts)
{
  splatt_free(opts);
}



splatt_kruskal * splatt_alloc_cpd(
    splatt_csf const * const csf,
    splatt_idx_t rank)
{
  splatt_kruskal * cpd = splatt_malloc(sizeof(*csf));

  cpd->lambda = splatt_malloc(rank * sizeof(*cpd->lambda));
  for(idx_t m=0; m < csf->nmodes; ++m) {
    cpd->factors[m] = splatt_malloc(csf->dims[m] * rank *
        sizeof(**cpd->factors));

    /* TODO: allow custom initialization including NUMA aware */
    fill_rand(cpd->factors[m], csf->dims[m] * rank);
  }

  return cpd;
}

#include "../io.h"




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


double cpd_iterate(
    splatt_csf const * const tensor,
    idx_t const rank,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts,
    splatt_kruskal * factored)
{
  idx_t const nmodes = tensor->nmodes;

  /* TODO: fix MTTKRP interface */
  matrix_t * mats[MAX_NMODES+1];
  for(idx_t m=0; m < tensor->nmodes; ++m) {
    mats[m] = mat_mkptr(factored->factors[m], tensor->dims[m], rank, 1);
  }
  mats[MAX_NMODES] = ws->mttkrp_buf;

  printf("CPD time\n");

  /* initialite aTa values */
  for(idx_t m=1; m < nmodes; ++m) {
    mat_aTa(mats[m], ws->aTa[m], NULL);
  }

  /* TODO: CSF opts */
  double * opts = splatt_default_opts();

  /* foreach outer iteration */
  for(idx_t it=0; it < cpd_opts->max_iterations; ++it) {
    /* foreach AO step */
    for(idx_t m=0; m < nmodes; ++m) {
      mttkrp_csf(tensor, mats, m, ws->thds, opts);

      mat_form_gram(ws->aTa, nmodes, m);

      /* ADMM solve for constraints */
      admm_inner(m, mats, ws, cpd_opts, global_opts);

      /* normalize columns and extract lambda */
      if(it == 0) {
        mat_normalize(mats[m], factored->lambda, MAT_NORM_2, NULL, ws->thds);
      } else {
        mat_normalize(mats[m], factored->lambda, MAT_NORM_MAX, NULL, ws->thds);
      }

      /* prepare aTa for next mode */
      mat_aTa(mats[m], ws->aTa[m], NULL);
    }
  }

  free(opts);

  for(idx_t m=0; m < tensor->nmodes; ++m) {
    splatt_free(mats[m]);
  }

  double fit = 1.;
  return fit;
}




cpd_ws * cpd_alloc_ws(
    splatt_csf const * const tensor,
    idx_t rank,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const nmodes = tensor->nmodes;

  cpd_ws * ws = splatt_malloc(sizeof(*ws));

  ws->nmodes = nmodes;
  for(idx_t m=0; m < nmodes; ++m) {
    ws->aTa[m] = mat_alloc(rank, rank);
  }

  ws->nthreads = global_opts->num_threads;
  ws->thds =  thd_init(ws->nthreads, 3,
    (rank * rank * sizeof(val_t)) + 64,
    0,
    (nmodes * rank * sizeof(val_t)) + 64);

  /* MTTKRP space */
  idx_t const maxdim = tensor->dims[argmax_elem(tensor->dims, nmodes)];
  ws->mttkrp_buf = mat_alloc(maxdim, rank);

  /* TODO: AO-ADMM constructs for constraints */

  return ws;
}



void cpd_free_ws(
    cpd_ws * const ws)
{
  mat_free(ws->mttkrp_buf);
  for(idx_t m=0; m < ws->nmodes; ++m) {
    mat_free(ws->aTa[m]);

    /* if constraints, free auxil/dual */
  }

  thd_free(ws->thds, ws->nthreads);
}


