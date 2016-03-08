

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
  val_t * directions[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    gradients[m] = splatt_malloc(model->dims[m] * model->rank *
        sizeof(**gradients));
    directions[m] = splatt_malloc(model->dims[m] * model->rank *
        sizeof(**directions));
  }

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  val_t prev_obj = loss + frobsq;
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  timer_start(&ws->tc_time);

  /* foreach epoch */
  for(idx_t e=1; e < ws->max_its+1; ++e) {

    tc_gradient(csf, model, ws, gradients);

    /* direction is the negative gradient */
    #pragma omp parallel
    {
      for(idx_t m=0; m < model->nmodes; ++m) {
        idx_t const N = model->dims[m] * model->rank;
        val_t const * const restrict grad = gradients[m];
        val_t * const restrict direc = directions[m];

        #pragma omp for schedule(static) nowait
        for(idx_t x=0; x < N; ++x) {
          direc[x] = -grad[x];
        }
      }
    }

    tc_line_search(train, model, ws, prev_obj, gradients, directions,
        &loss, &frobsq);
    prev_obj = loss + frobsq;


    printf("  time-grad: %0.3fs  time-line: %0.3fs\n",
        ws->grad_time.seconds, ws->line_time.seconds);
    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      break;
    }
  }

  for(idx_t m=0; m < nmodes; ++m) {
    splatt_free(gradients[m]);
    splatt_free(directions[m]);
  }

  csf_free(csf, opts);
  splatt_free_opts(opts);
}


