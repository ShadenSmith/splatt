

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
  val_t * const restrict hada  = ws->thds[tid].scratch[0];
  val_t * const restrict accum = ws->thds[tid].scratch[1];
  val_t * const restrict neqs  = ws->thds[tid].scratch[2];

  /* grab the top-level row */
  idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  val_t const * const restrict arow = avals + (a_id * nfactors);

  /* process each fiber */
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    val_t const * const restrict brow = bvals  + (fids[fib] * nfactors);

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[fib]+1; jj < fptr[fib+1]; ++jj) {
      val_t const v = vals[jj];
      val_t const * const restrict crow = cvals + (inds[jj] * nfactors);

      /* update gradients */
      for(idx_t r=0; r < nfactors; ++r) {
        accum[r] += v * cv[r];
        hada[r] = bv[r] * cv[r];
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

  timer_reset(&ws->train_time);
  timer_reset(&ws->test_time);

  /* foreach epoch */
  for(idx_t e=0; e < ws->max_its; ++e) {
    timer_start(&ws->train_time);

    /* process each slice */
    idx_t const nslices = csf->pt->nfibs[0];
    for(idx_t i=0; i < nslices; ++i) {
      p_process_tree(csf, i, model, ws);
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
  }

  csf_free(csf, opts);
}


