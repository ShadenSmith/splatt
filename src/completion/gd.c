

#include "completion.h"
#include "../csf.h"

#include <math.h>
#include <omp.h>




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


static void p_process_tree(
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
  val_t * const restrict gradA = ws->gradients[csf->dim_perm[0]];
  val_t * const restrict gradB = ws->gradients[csf->dim_perm[1]];
  val_t * const restrict gradC = ws->gradients[csf->dim_perm[2]];


  /* the row we're actually updating */
  idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  val_t * const restrict g_arow = gradA + (a_id * nfactors);

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

  /* grab the top-level row */
  val_t const * const restrict arow = avals + (a_id * nfactors);

  /* process each fiber */
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    val_t const * const restrict brow = bvals  + (fids[fib] * nfactors);
    val_t * const restrict g_brow = gradB + (fids[fib] * nfactors);

    /* push Hadmard products down tree */
    for(idx_t r=0; r < nfactors; ++r) {
      predict_buf[r] = arow[r] * brow[r];
    }

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj) {
      val_t const * const restrict crow = cvals + (inds[jj] * nfactors);
      val_t * const restrict g_crow = gradC + (inds[jj] * nfactors);

      /* compute the  predicted value and loss */
      val_t predicted = 0;
      for(idx_t r=0; r < nfactors; ++r) {
        predicted += crow[r] * predict_buf[r];
      }
      val_t const loss = vals[jj] - predicted;

      /* update gradients */
      for(idx_t r=0; r < nfactors; ++r) {
        g_arow[r] += (brow[r] * crow[r] * loss);
        g_brow[r] += (arow[r] * crow[r] * loss);
        g_crow[r] += (arow[r] * brow[r] * loss);
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

  /* permute regularization parameters based on CSF */
  val_t regularization[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    regularization[m] = ws->regularization[csf->dim_perm[m]];
  }

  val_t learn_rate = ws->learn_rate;
  val_t prev_obj = 0;
  val_t prev_val_rmse = 0;

  timer_reset(&ws->train_time);
  timer_reset(&ws->test_time);

  /* foreach epoch */
  for(idx_t e=0; e < ws->max_its; ++e) {
    timer_start(&ws->train_time);

    /* reset gradients */
    for(idx_t m=0; m < nmodes; ++m) {
      memset(ws->gradients[m], 0,
          train->dims[m] * model->rank * sizeof(**(ws->gradients)));
    }

    /* gradient computation -- process each slice */
    for(idx_t i=0; i < csf->pt->nfibs[0]; ++i) {
      p_process_tree(csf, i, model, ws);
    }

    /* now update model */
    for(idx_t m=0; m < nmodes; ++m) {
      val_t * const restrict mat = model->factors[m];
      val_t const * const restrict grad = ws->gradients[m];
      for(idx_t x=0; x < (train->dims[m] * model->rank); ++x) {
        mat[x] += learn_rate * (grad[x] - (regularization[m] * mat[x]));
      }
    }

    timer_stop(&ws->train_time);

    /* compute RMSE and adjust learning rate */
    timer_start(&ws->test_time);
    val_t const loss = tc_loss_sq(train, model, ws);
    val_t const frobsq = tc_frob_sq(model, ws);
    val_t const obj = loss + frobsq;
    val_t const train_rmse = sqrt(loss / train->nnz);
    val_t const val_rmse = tc_rmse(validate, model, ws);
    timer_stop(&ws->test_time);

    printf("epoch:%4"SPLATT_PF_IDX"   obj: %0.5e   "
        "RMSE-tr: %0.5e   RMSE-vl: %0.5e time-tr: %0.3fs  time-ts: %0.3fs\n",
        e+1, obj, train_rmse, val_rmse,
        ws->train_time.seconds, ws->test_time.seconds);

    /* adjust learning rate */
    if(e > 0) {
#if 0
      if(obj < prev_obj) {
        learn_rate *= 1.05;
      } else {
        learn_rate *= 0.50;
      }
#endif

      /* check convergence */
      if(fabs(val_rmse - prev_val_rmse) < 1e-8) {
        break;
      }
    }

    prev_obj = obj;
    prev_val_rmse = val_rmse;
  }

  /* save any changes */
  ws->learn_rate = learn_rate;

  csf_free(csf, opts);
}


