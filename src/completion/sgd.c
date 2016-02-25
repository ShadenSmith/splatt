

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "completion.h"
#include "../csf.h"
#include "../reorder.h"
#include "../util.h"
#include "../sort.h"

#include <math.h>
#include <omp.h>



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Update a three-mode model based on a given observation.
*
* @param train The training data.
* @param nnz_index The index of the observation to update from.
* @param model The model to update.
* @param ws Workspace to use.
*/
static inline void p_update_model3(
    sptensor_t const * const train,
    idx_t const nnz_index,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  idx_t const x = nnz_index;

  assert(train->nmodes == 3);

  idx_t * * const ind = train->ind;
  val_t * const restrict arow = model->factors[0] + (ind[0][x] * nfactors);
  val_t * const restrict brow = model->factors[1] + (ind[1][x] * nfactors);
  val_t * const restrict crow = model->factors[2] + (ind[2][x] * nfactors);

  /* predict value */
  val_t predicted = 0;
  for(idx_t f=0; f < nfactors; ++f) {
    predicted += arow[f] * brow[f] * crow[f];
  }
  val_t const loss = train->vals[x] - predicted;

  val_t const rate = ws->learn_rate;
  val_t const * const restrict reg = ws->regularization;

  /* update rows */
  for(idx_t f=0; f < nfactors; ++f) {
#if 0
    arow[f] += rate * ((loss * brow[f] * crow[f]) - (reg[0] * arow[f]));
    brow[f] += rate * ((loss * arow[f] * crow[f]) - (reg[1] * brow[f]));
    crow[f] += rate * ((loss * arow[f] * brow[f]) - (reg[2] * crow[f]));
#else
    val_t const moda = rate * ((loss * brow[f] * crow[f]) - (reg[0] * arow[f]));
    val_t const modb = rate * ((loss * arow[f] * crow[f]) - (reg[1] * brow[f]));
    val_t const modc = rate * ((loss * arow[f] * brow[f]) - (reg[2] * crow[f]));
    arow[f] += moda;
    brow[f] += modb;
    crow[f] += modc;
#endif
  }
}



static inline void p_update_model_csf3(
    splatt_csf const * const train,
    idx_t const i,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  csf_sparsity const * const pt = train->pt;
  assert(model->nmodes == 3);

  /* sparsity structure */
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];

  /* current model */
  val_t const * const restrict vals = pt->vals;
  val_t * const restrict avals = model->factors[train->dim_perm[0]];
  val_t * const restrict bvals = model->factors[train->dim_perm[1]];
  val_t * const restrict cvals = model->factors[train->dim_perm[2]];

  val_t const rate = ws->learn_rate;
  val_t const * const restrict reg = ws->regularization;

  /*
   * TODO: multithreading
   */
  int const tid = 0;
  /* thread buffers */
  val_t * const restrict predict_buf  = ws->thds[tid].scratch[0];

  /* grab the top-level row */
  idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  val_t * const restrict arow = avals + (a_id * nfactors);

  /* process each fiber */
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    val_t * const restrict brow = bvals + (fids[fib] * nfactors);

    /* push Hadmard products down tree */
    for(idx_t f=0; f < nfactors; ++f) {
      predict_buf[f] = arow[f] * brow[f];
    }

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj) {
      val_t * const restrict crow = cvals + (inds[jj] * nfactors);

      /* compute the  predicted value and loss */
      val_t predicted = 0;
      for(idx_t f=0; f < nfactors; ++f) {
        predicted += crow[f] * predict_buf[f];
      }
      val_t const loss = vals[jj] - predicted;

      /* update model */
      for(idx_t f=0; f < nfactors; ++f) {
        val_t const moda = rate * ((loss * brow[f] * crow[f]) - (reg[0] * arow[f]));
        val_t const modb = rate * ((loss * arow[f] * crow[f]) - (reg[1] * brow[f]));
        val_t const modc = rate * ((loss * arow[f] * brow[f]) - (reg[2] * crow[f]));
        //if( jj > 10  && jj < 20)
          //printf("MOD: %f %f %f\n", moda, modb, modc);
        arow[f] += moda;
        brow[f] += modb;
        crow[f] += modc;
      }
    }
  } /* foreach fiber */
}




/**
* @brief Update a model based on a given observation.
*
* @param train The training data.
* @param nnz_index The index of the observation to update from.
* @param model The model to update.
* @param ws Workspace to use.
*/
static void p_update_model(
    sptensor_t const * const train,
    idx_t const nnz_index,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nmodes = train->nmodes;
  if(nmodes == 3) {
    p_update_model3(train, nnz_index, model, ws);
    return;
  }

  idx_t const nfactors = model->rank;
  idx_t const x = nnz_index;

  val_t * const restrict buffer = ws->thds[omp_get_thread_num()].scratch[0];

  /* compute the error */
  val_t const err = train->vals[x] - tc_predict_val(model, train, x, buffer);

  idx_t * * const ind = train->ind;

  /* update each of the factor (row-wise) */
  for(idx_t m=0; m < nmodes; ++m) {

    /* first fill buffer with the Hadamard product of all rows but current */
    idx_t moff = (m + 1) % nmodes;
    val_t const * const restrict init_row = model->factors[moff] +
        (ind[moff][x] * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buffer[f] = init_row[f];
    }
    for(moff = 2; moff < nmodes; ++moff) {
      idx_t const madj = (m + moff) % nmodes;
      val_t const * const restrict row = model->factors[madj] +
          (ind[madj][x] * nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        buffer[f] *= row[f];
      }
    }

    /* now actually update the row */
    val_t * const restrict update_row = model->factors[m] +
        (ind[m][x] * nfactors);
    val_t const reg = ws->regularization[m];
    val_t const rate = ws->learn_rate;
    for(idx_t f=0; f < nfactors; ++f) {
      update_row[f] += rate * ((err * buffer[f]) - (reg * update_row[f]));
    }
  }
}







/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void splatt_tc_sgd(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  val_t const * const restrict train_vals = train->vals;

  /* convert training data to CSF-ALLMODE */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  splatt_csf * csf = csf_alloc(train, opts);
  assert(csf->ntiles == 1);

  idx_t * perm = splatt_malloc(train->nnz * sizeof(*perm));

  idx_t const nslices = csf[0].pt->nfibs[0];
  idx_t * perm_i = splatt_malloc(nslices * sizeof(*perm_i));

  /* init perm */
  for(idx_t n=0; n < train->nnz; ++n) {
    perm[n] = n;
  }
  for(idx_t n=0; n < nslices; ++n) {
    perm_i[n] = n;
  }

  timer_reset(&ws->train_time);
  timer_reset(&ws->test_time);

  tt_sort(train, 0, NULL);

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  /* for bold driver */
  val_t obj = loss + frobsq;
  val_t prev_obj = obj;

  /* foreach epoch */
  for(idx_t e=1; e < ws->max_its+1; ++e) {
    timer_start(&ws->train_time);

    /* new nnz ordering */

    /* update model from all training observations */
#if 0
    //shuffle_idx(perm, train->nnz);
    for(idx_t n=0; n < train->nnz; ++n) {
      p_update_model(train, n, model, ws);
    }
#else
    //shuffle_idx(perm_i, nslices);
    for(idx_t i=0; i < 2 /*nslices*/; ++i) {
      //printf("e %lu: %lu vs %lu\n", e, i, perm_i[i]);
      p_update_model_csf3(csf, i, model, ws);
    }
#endif
    timer_stop(&ws->train_time);

    /* compute RMSE and adjust learning rate */
    timer_start(&ws->test_time);
    loss = tc_loss_sq(train, model, ws);
    frobsq = tc_frob_sq(model, ws);
    obj = loss + frobsq;
    timer_stop(&ws->test_time);
    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      break;
    }

    /* bold driver */
    if(e > 1) {
      if(obj < prev_obj) {
        ws->learn_rate *= 1.05;
      } else {
        ws->learn_rate *= 0.50;
      }
    }

    prev_obj = obj;
  }

  splatt_free(perm);
  splatt_free(perm_i);
  csf_free(csf, opts);
  splatt_free_opts(opts);
}


