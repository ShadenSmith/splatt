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
#include <math.h>

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
    splatt_cpd_opts const * const cpd_options,
    splatt_global_opts const * const global_options,
    splatt_kruskal * factored)
{
  /* allocate default options if they were not supplied */
  splatt_global_opts * global_opts = (splatt_global_opts *) global_options;
  if(global_options == NULL) {
    global_opts = splatt_alloc_global_opts();
  }
  splatt_cpd_opts * cpd_opts = (splatt_cpd_opts *) cpd_options;
  if(cpd_options == NULL) {
    cpd_opts = splatt_alloc_cpd_opts();
  }

  splatt_omp_set_num_threads(global_opts->num_threads);

  /* allocate workspace */
  cpd_ws * ws = cpd_alloc_ws(tensor, rank, cpd_opts, global_opts);

  /* perform the factorization! */
  cpd_iterate(tensor, rank, ws, cpd_opts, global_opts, factored);

  /* clean up workspace */
  cpd_free_ws(ws);

  /* free options if we had to allocate them */
  if(global_options == NULL) {
    splatt_free_global_opts(global_opts);
  }
  if(cpd_options == NULL) {
    splatt_free_cpd_opts(cpd_opts);
  }

  return SPLATT_SUCCESS;
}



splatt_cpd_opts * splatt_alloc_cpd_opts(void)
{
  splatt_cpd_opts * opts = splatt_malloc(sizeof(*opts));

  /* defaults */
  opts->tolerance = 1e-5;
  opts->max_iterations = 200;

  opts->inner_tolerance = 1e-2;
  opts->max_inner_iterations = 20;

  for(idx_t m=0; m < MAX_NMODES; ++m) {
    opts->constraints[m].which = SPLATT_CON_NONE;
    opts->constraints[m].data = NULL;
  }

  return opts;
}


void splatt_free_cpd_opts(
    splatt_cpd_opts * opts)
{
  /* if constraints, free data */
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    switch(opts->constraints[m].which) {
    case SPLATT_CON_NONE:
      break;
    case SPLATT_REG_L1:
      splatt_free(opts->constraints[m].data);
      break;
    case SPLATT_REG_L2:
      splatt_free(opts->constraints[m].data);
      break;
    case SPLATT_CON_NONNEG:
      break;
    }
  }

  /* free options pointer */
  splatt_free(opts);
}


void splatt_cpd_reg_l1(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale)
{
  /* MAX_NMODES will simply apply regularization to all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_reg_l1(cpd_opts, m, scale);
    }
    return;
  }

  splatt_cpd_con_clear(cpd_opts, mode);

  cpd_opts->constraints[mode].which = SPLATT_REG_L1;
  val_t * lambda = splatt_malloc(sizeof(*lambda));
  *lambda = scale;
  cpd_opts->constraints[mode].data = (void *) lambda;
}

void splatt_cpd_reg_l2(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale)
{
  /* MAX_NMODES will simply apply regularization to all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_reg_l2(cpd_opts, m, scale);
    }
    return;
  }

  splatt_cpd_con_clear(cpd_opts, mode);

  cpd_opts->constraints[mode].which = SPLATT_REG_L2;
  val_t * lambda = splatt_malloc(sizeof(*lambda));
  *lambda = scale;
  cpd_opts->constraints[mode].data = (void *) lambda;
}


void splatt_cpd_con_nonneg(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode)
{
  /* MAX_NMODES will simply apply constraints to all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_con_nonneg(cpd_opts, m);
    }
    return;
  }

  splatt_cpd_con_clear(cpd_opts, mode);

  cpd_opts->constraints[mode].which = SPLATT_CON_NONNEG;
}


void splatt_cpd_con_clear(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode)
{
  /* MAX_NMODES will operate on all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_con_clear(cpd_opts, m);
    }
    return;
  }

  cpd_opts->constraints[mode].which = SPLATT_CON_NONE;
  if(cpd_opts->constraints[mode].data != NULL) {
    splatt_free(cpd_opts->constraints[mode].data);
    cpd_opts->constraints[mode].data = NULL;
  }
}


splatt_kruskal * splatt_alloc_cpd(
    splatt_csf const * const csf,
    splatt_idx_t rank)
{
  splatt_kruskal * cpd = splatt_malloc(sizeof(*csf));

  cpd->nmodes = csf->nmodes;

  cpd->lambda = splatt_malloc(rank * sizeof(*cpd->lambda));
  for(idx_t m=0; m < csf->nmodes; ++m) {
    cpd->factors[m] = splatt_malloc(csf->dims[m] * rank *
        sizeof(**cpd->factors));

    /* TODO: allow custom initialization including NUMA aware */
    fill_rand(cpd->factors[m], csf->dims[m] * rank);
  }

  /* initialize lambda in case it is not modified */
  for(idx_t r=0; r < rank; ++r) {
    cpd->lambda[r] = 1.;
  }

  return cpd;
}



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

    mat_normalize(mats[m], factored->lambda, MAT_NORM_2, NULL, ws->thds);
  }
  mats[MAX_NMODES] = ws->mttkrp_buf;

  /* reset column weights */
  for(idx_t r=0; r < rank; ++r) {
    factored->lambda[r] = 1.;
  }

  /* initialite aTa values */
  for(idx_t m=1; m < nmodes; ++m) {
    mat_aTa(mats[m], ws->aTa[m], NULL);
  }

  /* TODO: CSF opts */
  double * opts = splatt_default_opts();

  /* for tracking convergence */
  double oldfit = 0.;
  double fit = 0.;
  double const ttnormsq = csf_frobsq(tensor);

  /* timers */
  sp_timer_t itertime;
  sp_timer_t modetime[MAX_NMODES];
  timer_start(&timers[TIMER_CPD]);

  idx_t inner_its[MAX_NMODES];

  /* foreach outer iteration */
  for(idx_t it=0; it < cpd_opts->max_iterations; ++it) {
    timer_fstart(&itertime);
    /* foreach AO step */
    for(idx_t m=0; m < nmodes; ++m) {
      timer_fstart(&modetime[m]);
      mttkrp_csf(tensor, mats, m, ws->thds, opts);

      /* ADMM solve for constraints */
      inner_its[m] = admm_inner(m, mats, factored->lambda, ws, cpd_opts,
          global_opts);

      /* prepare aTa for next mode */
      mat_aTa(mats[m], ws->aTa[m], NULL);

      timer_stop(&modetime[m]);
    } /* foreach mode */

    /* calculate outer convergence */
    double const norm = cpd_norm(ws, factored->lambda);
    double const inner = cpd_innerprod(nmodes-1, ws, mats, factored->lambda);
    double const residual = sqrt(ttnormsq + norm - (2 * inner));
    fit = 1 - (residual / sqrt(ttnormsq));

    assert(fit >= oldfit);
    timer_stop(&itertime);

    /* print progress */
    if(global_opts->verbosity > SPLATT_VERBOSITY_NONE) {
      printf("  its = %3"SPLATT_PF_IDX" (%0.3"SPLATT_PF_VAL"s)  "
             "fit = %0.5"SPLATT_PF_VAL"  delta = %+0.4e\n",
          it+1, itertime.seconds, fit, fit - oldfit);
      if(global_opts->verbosity > SPLATT_VERBOSITY_LOW) {
        for(idx_t m=0; m < nmodes; ++m) {
          printf("     mode = %1"SPLATT_PF_IDX" (%0.3fs) "
                 "[%3"SPLATT_PF_IDX" ADMM its]\n", m+1,
              modetime[m].seconds, inner_its[m]);
        }
      }
    }

    /* terminate if converged */
    if(it > 0 && fabs(fit - oldfit) < cpd_opts->tolerance) {
      break;
    }
    oldfit = fit;
  }

#if 0
  cpd_post_process(mats, factored->lambda, ws, cpd_opts, global_opts);
#endif

  splatt_free(opts);
  for(idx_t m=0; m < tensor->nmodes; ++m) {
    /* only free ptr */
    splatt_free(mats[m]);
  }

  timer_stop(&timers[TIMER_CPD]);

  return fit;
}



void cpd_post_process(
    matrix_t * * mats,
    val_t * const column_weights,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const rank = mats[0]->J;
  val_t * tmp =  splatt_malloc(rank * sizeof(*tmp));

  /* normalize each matrix and adjust lambda */
  for(idx_t m=0; m < ws->nmodes; ++m) {
    mat_normalize(mats[m], tmp, MAT_NORM_2, NULL, ws->thds);
    for(idx_t f=0; f < rank; ++f) {
      column_weights[f] *= tmp[f];
    }
  }

  splatt_free(tmp);
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
  ws->aTa_buf = mat_alloc(rank, rank);
  ws->gram = mat_alloc(rank, rank);

  ws->nthreads = global_opts->num_threads;
  ws->thds =  thd_init(ws->nthreads, 3,
    (rank * rank * sizeof(val_t)) + 64,
    0,
    (nmodes * rank * sizeof(val_t)) + 64);

  /* MTTKRP space */
  idx_t const maxdim = tensor->dims[argmax_elem(tensor->dims, nmodes)];
  ws->mttkrp_buf = mat_alloc(maxdim, rank);


  /* AO-ADMM */
  ws->auxil    = mat_alloc(maxdim, rank);
  ws->mat_init = mat_alloc(maxdim, rank);
  for(idx_t m=0; m < nmodes; ++m) {
    ws->duals[m] = mat_alloc(tensor->dims[m], rank);

    /* duals should be 0 */
    memset(ws->duals[m]->vals, 0, tensor->dims[m] * rank * sizeof(val_t));
  }

  return ws;
}



void cpd_free_ws(
    cpd_ws * const ws)
{
  mat_free(ws->mttkrp_buf);
  mat_free(ws->aTa_buf);
  mat_free(ws->gram);

  mat_free(ws->auxil);
  mat_free(ws->mat_init);

  for(idx_t m=0; m < ws->nmodes; ++m) {
    mat_free(ws->aTa[m]);
    mat_free(ws->duals[m]);
  }

  thd_free(ws->thds, ws->nthreads);
  splatt_free(ws);
}


val_t cpd_norm(
    cpd_ws const * const ws,
    val_t const * const restrict column_weights)
{
  idx_t const rank = ws->aTa[0]->J;
  val_t * const restrict scratch = ws->aTa_buf->vals;

  /* initialize scratch space */
  for(idx_t i=0; i < rank; ++i) {
    for(idx_t j=i; j < rank; ++j) {
      scratch[j + (i*rank)] = 1.;
    }
  }

  /* scratch = hada(aTa) */
  for(idx_t m=0; m < ws->nmodes; ++m) {
    val_t const * const restrict atavals = ws->aTa[m]->vals;
    for(idx_t i=0; i < rank; ++i) {
      for(idx_t j=i; j < rank; ++j) {
        scratch[j + (i*rank)] *= atavals[j + (i*rank)];
      }
    }
  }

  /* now compute weights^T * aTa[MAX_NMODES] * weights */
  val_t norm = 0;
  for(idx_t i=0; i < rank; ++i) {
    norm += scratch[i+(i*rank)] * column_weights[i] * column_weights[i];
    for(idx_t j=i+1; j < rank; ++j) {
      norm += scratch[j+(i*rank)] * column_weights[i] * column_weights[j] * 2;
    }
  }

  return fabs(norm);
}


val_t cpd_innerprod(
    idx_t lastmode,
    cpd_ws const * const ws,
    matrix_t * * mats,
    val_t const * const restrict column_weights)
{
  idx_t const nrows = mats[lastmode]->I;
  idx_t const rank = mats[0]->J;

  val_t const * const newmat = mats[lastmode]->vals;
  val_t const * const mttkrp = ws->mttkrp_buf->vals;

  val_t myinner = 0;
  #pragma omp parallel reduction(+:myinner)
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * const restrict accumF = ws->thds[tid].scratch[0];

    for(idx_t r=0; r < rank; ++r) {
      accumF[r] = 0.;
    }

    /* Hadamard product with newest factor and previous MTTKRP */
    #pragma omp for schedule(static)
    for(idx_t i=0; i < nrows; ++i) {
      val_t const * const restrict newmat_row = newmat + (i*rank);
      val_t const * const restrict mttkrp_row = mttkrp + (i*rank);
      for(idx_t r=0; r < rank; ++r) {
        accumF[r] += newmat_row[r] * mttkrp_row[r];
      }
    }

    /* accumulate everything into 'myinner' */
    for(idx_t r=0; r < rank; ++r) {
      myinner += accumF[r] * column_weights[r];
    }
  } /* end omp parallel -- reduce myinner */

  /* TODO AllReduce for MPI support */

  return myinner;
}

