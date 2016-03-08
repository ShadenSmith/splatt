
#include "completion.h"
#include "gradient.h"
#include "../csf.h"
#include "../util.h"

#include <math.h>
#include <omp.h>



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


static void p_update_direction(
    tc_model const * const model,
    val_t const beta,
    val_t * * gradients,
    val_t * * directions)
{
  /* Restart, just go for steepest descent (negative gradient).
   * NOTE: we don't just run the normal code (multiplying by beta=0) because
   * the first iteration doesn't initialize directions.
   */
  if(beta == 0) {
    #pragma omp parallel
    {
      for(idx_t m=0; m < model->nmodes; ++m) {
        idx_t const N = model->dims[m] * model->rank;

        val_t * const restrict dir = directions[m];
        val_t const * const restrict grad = gradients[m];
        #pragma omp for schedule(static) nowait
        for(idx_t x=0; x < N; ++x) {
          dir[x] = -grad[x];
        }
      }
    } /* omp parallel */
    return;
  }

  #pragma omp parallel
  {
    for(idx_t m=0; m < model->nmodes; ++m) {
      idx_t const N = model->dims[m] * model->rank;

      val_t * const restrict dir = directions[m];
      val_t const * const restrict grad = gradients[m];
      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < N; ++x) {
        /* p_{k+1} \gets - gradient + \Beta p_{k} */
        dir[x] = -grad[x] + (beta * dir[x]);
      }
    }
  } /* omp parallel */
}



/**
* @brief Choose \beta using the Polak-Ribiere method with automatic restarts.
*
* @param model The current model. Just used for dimensionality.
* @param prev_gradients The gradient of the previous iteration.
* @param gradients The latest gradient.
*
* @return Beta, used to scale the previous direction when choosing the new
*         one. When a restart is needed, beta=0 and we take a steepest descent
*         step.
*/
static val_t p_conjugate_prplus(
    tc_model const * const model,
    val_t * * prev_gradients,
    val_t * * gradients)
{
  val_t numer = 0; /* numerator is grad^T (grad - prev_grad) */
  val_t denom = 0; /* denominator is || prev_grad ||^2 */

  #pragma omp parallel reduction(+:numer, denom)
  {
    for(idx_t m=0; m < model->nmodes; ++m) {
      idx_t const N = model->dims[m] * model->rank;
      val_t const * const restrict grad = gradients[m];
      val_t const * const restrict pgrad = prev_gradients[m];

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < N; ++x) {
        numer += grad[x] * (grad[x] - pgrad[x]);
        denom += pgrad[x] * pgrad[x];
      }
    }
  } /* omp parallel */

  /* beta <= 0 -> restart and use steepest descent */
  val_t const beta = numer / denom;
  return SS_MAX(0, beta);
}



/**
* @brief Choose \beta using the Fletcher-Reeves method.
*
* @param model The current model. Just used for dimensionality.
* @param prev_gradients The gradient of the previous iteration.
* @param gradients The latest gradient.
*
* @return Beta, used to scale the previous direction when choosing the new
*         one.
*/
static val_t p_conjugate_fr(
    tc_model const * const model,
    val_t * * prev_gradients,
    val_t * * gradients)
{
  val_t numer = 0; /* numerator is || grad ||^2 */
  val_t denom = 0; /* denominator is || prev_grad ||^2 */

  #pragma omp parallel reduction(+:numer, denom)
  {
    for(idx_t m=0; m < model->nmodes; ++m) {
      idx_t const N = model->dims[m] * model->rank;

      val_t const * const restrict grad = gradients[m];
      val_t const * const restrict pgrad = prev_gradients[m];
      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < N; ++x) {
        numer += grad[x] * grad[x];
        denom += pgrad[x] * pgrad[x];
      }
    }
  } /* omp parallel */

  return numer / denom;
}



/**
* @brief Swap the previous and current gradients. The pointers are swapped.
*
* @param nmodes The number of gradients.
* @param[out] prev_grads The gradients of the previous iterations.
* @param[out] grads The latest gradients.
*/
static void p_swap_gradients(
    idx_t const nmodes,
    val_t * * prev_grads,
    val_t * * grads)
{
  for(idx_t m=0; m < nmodes; ++m) {
    val_t * tmp = prev_grads[m];
    prev_grads[m] = grads[m];
    grads[m] = tmp;
  }
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void splatt_tc_nlcg(
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

  /* allocate gradients */
  val_t * directions[MAX_NMODES];
  val_t * gradients[MAX_NMODES];
  val_t * prev_gradients[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    size_t const bytes = model->dims[m] * model->rank * sizeof(**gradients);
    directions[m]     = splatt_malloc(bytes);
    gradients[m]      = splatt_malloc(bytes);
    prev_gradients[m] = splatt_malloc(bytes);
  }

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  val_t prev_obj = loss + frobsq;
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  timer_start(&ws->tc_time);

  /* initialize direction with negative gradient */
  tc_gradient(csf, model, ws, gradients);
  p_update_direction(model, 0, gradients, directions);

  /* foreach epoch */
  for(idx_t e=1; e < ws->max_its+1; ++e) {

    /* find the new iterate */
    tc_line_search(train, model, ws, prev_obj, gradients, directions,
        &loss, &frobsq);
    prev_obj = loss + frobsq;

    /* check for convergence */
    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      break;
    }

    /* not converged, find next direction */
    p_swap_gradients(nmodes, prev_gradients, gradients);
    tc_gradient(csf, model, ws, gradients);
    val_t const beta = p_conjugate_prplus(model, prev_gradients, gradients);
    p_update_direction(model, beta, gradients, directions);


    printf("  time-grad: %0.3fs  time-line: %0.3fs\n",
        ws->grad_time.seconds, ws->line_time.seconds);
  }

  for(idx_t m=0; m < nmodes; ++m) {
    splatt_free(directions[m]);
    splatt_free(gradients[m]);
    splatt_free(prev_gradients[m]);
  }

  csf_free(csf, opts);
  splatt_free_opts(opts);
}


