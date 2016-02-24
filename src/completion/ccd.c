
/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "completion.h"
#include "../util.h"

#include <math.h>
#include <omp.h>



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void splatt_tc_ccd(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws)
{

  timer_reset(&ws->train_time);
  timer_reset(&ws->test_time);

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  /* foreach epoch */
  for(idx_t e=1; e < ws->max_its+1; ++e) {
    /* update model from all training observations */
    timer_start(&ws->train_time);

    timer_stop(&ws->train_time);

    /* compute RMSE and adjust learning rate */
    timer_start(&ws->test_time);
    loss = tc_loss_sq(train, model, ws);
    frobsq = tc_frob_sq(model, ws);
    timer_stop(&ws->test_time);
    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      break;
    }

  } /* foreach epoch */
}
