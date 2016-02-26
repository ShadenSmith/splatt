

#include "completion.h"
#include "../csf.h"

#include <math.h>
#include <omp.h>




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Update the TC model using ws->gradients and learn_rate. Also reverses
*        a previous step of size prev_rate. This lets us just use one update
*        during a line search.
*
* @param model The TC model to update.
* @param ws Workspace storing gradients.
* @param prev_rate Previous step to reverse. 0 is safe to use.
* @param learn_rate The new step size.
*/
static void p_update_model(
    tc_model * const model,
    tc_ws * const ws,
    val_t const prev_rate,
    val_t const learn_rate)
{
  val_t const new_rate = learn_rate - prev_rate;

  #pragma omp parallel
  {
    for(idx_t m=0; m < model->nmodes; ++m) {
      val_t * const restrict mat = model->factors[m];
      val_t const * const restrict grad = ws->gradients[m];
      val_t const reg = ws->regularization[m];

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < (model->dims[m] * model->rank); ++x) {
        mat[x] += new_rate * grad[x];
      }
    }
  } /* end omp parallel */
}



/**
* @brief Perform a simple backtracking line search to find a new step size that
*        reduces the objective function.
*
* @param train The training data for evaluating objective.
* @param model The model to update.
* @param ws Workspace storing gradients.
* @param prev_obj The previous objective value.
* @param[out] ret_loss On exit, stores the new loss value.
* @param[out] ret_frobsq On exit, stores the new \sum \lambda||A||_F^2 penalty.
*/
static void p_line_search(
    sptensor_t const * const train,
    tc_model * const model,
    tc_ws * const ws,
    val_t const prev_obj,
    val_t * ret_loss,
    val_t * ret_frobsq)
{
  val_t learn_rate = 5e-4;
  val_t loss;
  val_t frobsq;
  idx_t neval = 1;

  /* update model */
  p_update_model(model, ws, 0, learn_rate);

  while(true) {
    /* compute new objective */
    loss = tc_loss_sq(train, model, ws);
    frobsq = tc_frob_sq(model, ws);
    val_t const obj = loss + frobsq;

    if(obj < prev_obj) {
#if 0
      printf("  %0.5f -> %0.5f (neval: %lu learn: %0.3e)\n",
          prev_obj, obj, neval, learn_rate);
#endif
      break;
    }

    /* change learning rate and update model (while undoing last step) */
    p_update_model(model, ws, learn_rate, learn_rate * 0.5);
    learn_rate *= 0.50;

    ++neval;
  }

  *ret_loss = loss;
  *ret_frobsq = frobsq;
}




/**
* @brief Update all gradient matrices based on the observations in tree i.
*
* @param csf The training data.
* @param i Which tree to process.
* @param model The model to update
* @param ws Workspace. ws->gradients are accessed.
*/
static void p_process_tree3(
    splatt_csf const * const csf,
    idx_t const i,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  csf_sparsity const * const pt = csf->pt;
  assert(model->nmodes == 3);

  /* sparsity structure */
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];

  /* gradients */
  val_t * const restrict grad_a = ws->gradients[csf->dim_perm[0]];
  val_t * const restrict grad_b = ws->gradients[csf->dim_perm[1]];
  val_t * const restrict grad_c = ws->gradients[csf->dim_perm[2]];

  /* current model */
  val_t const * const restrict vals = pt->vals;
  val_t const * const restrict avals = model->factors[csf->dim_perm[0]];
  val_t const * const restrict bvals = model->factors[csf->dim_perm[1]];
  val_t const * const restrict cvals = model->factors[csf->dim_perm[2]];

  /*
   * TODO: multithreading
   */
  int const tid = 0;
  /* thread buffers */
  val_t * const restrict predict_buf  = ws->thds[tid].scratch[0];

  /* the row we're actually updating */
  idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];

  /* grab the top-level row */
  val_t const * const restrict arow = avals + (a_id * nfactors);
  val_t * const restrict g_arow = grad_a + (a_id * nfactors);

  /* process each fiber */
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    val_t const * const restrict brow = bvals  + (fids[fib] * nfactors);
    val_t * const restrict g_brow = grad_b + (fids[fib] * nfactors);

    /* push Hadmard products down tree */
    for(idx_t f=0; f < nfactors; ++f) {
      predict_buf[f] = arow[f] * brow[f];
    }

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj) {
      val_t const * const restrict crow = cvals + (inds[jj] * nfactors);
      val_t * const restrict g_crow = grad_c + (inds[jj] * nfactors);

      /* compute the loss */
      val_t loss = vals[jj];
      for(idx_t f=0; f < nfactors; ++f) {
        loss -= predict_buf[f] * crow[f];
      }

      /* update gradients */
      for(idx_t f=0; f < nfactors; ++f) {
        g_arow[f] += brow[f] * crow[f] * loss;
        g_brow[f] += arow[f] * crow[f] * loss;
        g_crow[f] += arow[f] * brow[f] * loss;
      }
    }
  } /* foreach fiber */
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void splatt_tc_gd(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws)
{
  /* convert training data to CSF-ALLMODE */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  splatt_csf * csf = csf_alloc(train, opts);
  assert(csf->ntiles == 1);

  idx_t const nmodes = train->nmodes;


  timer_reset(&ws->train_time);
  timer_reset(&ws->test_time);
  timer_reset(&ws->grad_time);
  timer_reset(&ws->line_time);

  val_t learn_rate = ws->learn_rate;

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  val_t prev_obj = loss + frobsq;
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  /* foreach epoch */
  for(idx_t e=1; e < ws->max_its+1; ++e) {
    timer_start(&ws->train_time);

    timer_start(&ws->grad_time);
    #pragma omp parallel
    {
      /* reset gradients - initialize with negative regularization */
      for(idx_t m=0; m < nmodes; ++m) {
        val_t const * const restrict mat = model->factors[m];
        val_t * const restrict grad = ws->gradients[m];
        val_t const reg = ws->regularization[m];

        #pragma omp for schedule(static) nowait
        for(idx_t x=0; x < (train->dims[m] * model->rank); ++x) {
          grad[x] = -(reg * mat[x]);
        }
      }
    }

    /* gradient computation -- process each slice */
    for(idx_t i=0; i < csf->pt->nfibs[0]; ++i) {
      p_process_tree3(csf, i, model, ws);
    }
    timer_stop(&ws->grad_time);

    timer_start(&ws->line_time);
    p_line_search(train, model, ws, prev_obj, &loss, &frobsq);
    timer_stop(&ws->line_time);

    prev_obj = loss + frobsq;

    timer_stop(&ws->train_time);

    printf("  time-grad: %0.3fs  time-line: %0.3fs\n",
        ws->grad_time.seconds, ws->line_time.seconds);
    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      break;
    }
  }

  /* save any changes */
  ws->learn_rate = learn_rate;

  csf_free(csf, opts);
  splatt_free_opts(opts);
}


