
/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "gradient.h"



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Update the TC model using directions and learn_rate. Also reverses
*        a previous step of size prev_rate. This lets us just use one update
*        during a line search.
*
* @param model The TC model to update.
* @param ws Workspace storing gradients.
* @param directions Directions for each factor.
* @param prev_rate Previous step to reverse. 0 is safe to use.
* @param learn_rate The new step size.
*/
static void p_update_model(
    tc_model * const model,
    tc_ws * const ws,
    val_t * * directions,
    val_t const prev_rate,
    val_t const learn_rate)
{
  val_t const new_rate = learn_rate - prev_rate;

  #pragma omp parallel
  {
    for(idx_t m=0; m < model->nmodes; ++m) {
      val_t * const restrict mat = model->factors[m];
      val_t const * const restrict direc = directions[m];

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < (model->dims[m] * model->rank); ++x) {
        mat[x] += new_rate * direc[x];
      }
    }
  } /* end omp parallel */
}


/**
* @brief Update all gradient matrices based on the observations in tree i.
*
* @param csf The training data.
* @param i Which tree to process.
* @param model The model to update
* @param ws Workspace.
* @param[out] gradients Gradients to compute.
*/
static void p_process_tree3(
    splatt_csf const * const csf,
    idx_t const i,
    tc_model const * const model,
    tc_ws * const ws,
    val_t * * gradients)
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
  val_t * const restrict grad_a = gradients[csf->dim_perm[0]];
  val_t * const restrict grad_b = gradients[csf->dim_perm[1]];
  val_t * const restrict grad_c = gradients[csf->dim_perm[2]];

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

void tc_gradient(
    splatt_csf const * const train,
    tc_model const * const model,
    tc_ws * const ws,
    val_t * * gradients)
{
  timer_start(&ws->grad_time);

  /* reset gradients */
  #pragma omp parallel
  {
    for(idx_t m=0; m < model->nmodes; ++m) {
      val_t * const restrict grad = gradients[m];

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < (train->dims[m] * model->rank); ++x) {
        grad[x] = 0;
      }
    }
  }


  /* gradient computation -- process each slice */
  for(idx_t i=0; i < train->pt->nfibs[0]; ++i) {
    p_process_tree3(train, i, model, ws, gradients);
  }

  /* Gradient is actually -grad + \lambda A */

  #pragma omp parallel
  {
    for(idx_t m=0; m < model->nmodes; ++m) {
      val_t const * const restrict mat = model->factors[m];
      val_t * const restrict grad = gradients[m];
      val_t const reg = ws->regularization[m];

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < (train->dims[m] * model->rank); ++x) {
        grad[x] = -grad[x] + (reg * mat[x]);
      }
    }
  }

  timer_stop(&ws->grad_time);
}


/**
* @brief Compute the dot product between the gradient and the new direction.
*
* @param model Just for vector sizes.
* @param gradients Gradient matrices (non-negated).
* @param directions The new direction vector.
*
* @return gradient^T direction.
*/
static val_t p_grad_dot(
    tc_model const * const model,
    val_t * * gradients,
    val_t * * directions)
{
  val_t grad_dot = 0;

  #pragma omp parallel reduction(+:grad_dot)
  {
    for(idx_t m=0; m < model->nmodes; ++m) {
      val_t const * const grad = gradients[m];
      val_t const * const direc = directions[m];
      idx_t const N = model->dims[m] * model->rank;

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < N; ++x) {
        grad_dot += grad[x] * direc[x];
      }
    }
  }

  return grad_dot;
}



void tc_line_search(
    sptensor_t const * const train,
    tc_model * const model,
    tc_ws * const ws,
    val_t const prev_obj,
    val_t * * gradients,
    val_t * * directions,
    val_t * ret_loss,
    val_t * ret_frobsq)
{
  timer_start(&ws->line_time);
  val_t loss;
  val_t frobsq;
  idx_t neval = 0;

  val_t learn_rate = ws->learn_rate * 4;
  val_t const dec = 0.5;

  /* Wolfe conditions TODO: choose better? */
  val_t const wolfe = 1e-5;

  val_t gdot = p_grad_dot(model, gradients, directions);
  if(gdot > 0) {
    return;
  }

  /* update model with initial step size */
  p_update_model(model, ws, directions, 0, learn_rate);

  while(true) {
    ++neval;

    /* compute new objective */
    loss = tc_loss_sq(train, model, ws);
    frobsq = tc_frob_sq(model, ws);
    val_t const obj = loss + frobsq;

    if(obj < prev_obj + (learn_rate * wolfe * gdot)) {
      break;
    }

    /* change learning rate and update model (while undoing last step) */
    p_update_model(model, ws, directions, learn_rate, learn_rate * dec);
    learn_rate *= dec;

    if(neval > 25) {
      printf("  LINE SEARCH STALLED, >25 EVALUATIONS.\n");
      break;
    }
  }

  timer_stop(&ws->line_time);
#if 0
  printf("  %0.5e -> %0.5e delta: %0.5e (neval: %lu learn: %0.3e)\n",
      prev_obj, loss + frobsq, loss + frobsq - prev_obj, neval-1, learn_rate);
#endif

  ws->learn_rate = learn_rate;
  *ret_loss = loss;
  *ret_frobsq = frobsq;
}


