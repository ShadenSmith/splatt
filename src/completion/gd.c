

#include "completion.h"
#include "gradient.h"
#include "../csf.h"

#include <math.h>
#include <omp.h>




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/



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

  /* allocate gradients */
  val_t * gradients[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    gradients[m] = splatt_malloc(train->dims[m] * model->rank *
        sizeof(**gradients));
  }

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

    tc_gradient(csf, model, ws, gradients);

    tc_line_search(train, model, ws, prev_obj, gradients, gradients,
        &loss, &frobsq);

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

  for(idx_t m=0; m < nmodes; ++m) {
    splatt_free(gradients[m]);
  }

  csf_free(csf, opts);
  splatt_free_opts(opts);
}


