
#include "sgd.h"
#include "reorder.h"
#include "util.h"

#include <math.h>



/**
* @brief Compute the value of the objective function:
*        \sum_{nnz \in train} (train(x) - predict(x))^2  +
*        \sum_{m=1}^{nmodes} reg[m] ||factors[m]||_F^2
*
* @param train The training data used to compute loss.
* @param model The model we are evaluating.
* @param regularization The regularization parameters.
*
* @return The value of the objective function.
*/
static val_t p_calc_obj(
    sptensor_t const * const train,
    splatt_kruskal const * const model,
    val_t const * const restrict regularization)
{
  idx_t const nfactors = model->rank;
  idx_t const nmodes = train->nmodes;

  val_t reg_obj = 0;
  val_t loss_obj = 0;

  #pragma omp parallel reduction(+:reg_obj,loss_obj)
  {
    val_t * buffer = splatt_malloc(nfactors * sizeof(*buffer));

    for(idx_t m=0; m < nmodes; ++m) {
      val_t accum = 0;
      val_t const * const restrict mat = model->factors[m];
      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < model->dims[m] * nfactors; ++x) {
        accum += mat[x] * mat[x];
      }
      reg_obj += regularization[m] * accum;
    }

    val_t const * const restrict train_vals = train->vals;
    #pragma omp for schedule(static) nowait
    for(idx_t x=0; x < train->nnz; ++x) {
      val_t const err = train_vals[x] - predict_val(model, train, x, buffer);
      loss_obj += err * err;
    }

    splatt_free(buffer);
  }

  return loss_obj + reg_obj;
}


/**
* @brief Update a model based on a given observation.
*
* @param train The training data.
* @param nnz_index The index of the observation to update from.
* @param model The model to update.
* @param buffer A buffer of size nfactors.
* @param learn_rate The step size to take in the update.
* @param regularization Regularization parameters.
*/
static void p_update_model(
    sptensor_t const * const train,
    idx_t const nnz_index,
    splatt_kruskal const * const model,
    val_t * const buffer,
    val_t const learn_rate,
    val_t const * const regularization)
{
  idx_t const nfactors = model->rank;
  idx_t const nmodes = train->nmodes;
  idx_t const x = nnz_index;

  /* compute the error */
  val_t const err = train->vals[x] - predict_val(model, train, x, buffer);

  for(idx_t m=0; m < nmodes; ++m) {
    for(idx_t f=0; f < nfactors; ++f) {
      buffer[f] = 1.;
    }

    for(idx_t m2=0; m2 < nmodes; ++m2) {
      if(m2 != m) {
        val_t const * const restrict row = model->factors[m2] +
            (train->ind[m2][x] * nfactors);
        for(idx_t f=0; f < nfactors; ++f) {
          buffer[f] *= row[f];
        }
      }
    }

    val_t * const restrict update_row = model->factors[m] +
        (train->ind[m][x] * nfactors);
    val_t const reg = regularization[m];
    for(idx_t f=0; f < nfactors; ++f) {
      update_row[f] += learn_rate * ((err * buffer[f]) - (reg * update_row[f]));
    }
  }
}


void splatt_sgd(
    sptensor_t const * const train,
    sptensor_t const * const validate,
    splatt_kruskal * const model,
    idx_t const max_epochs,
    val_t learn_rate,
    val_t const * const regularization)
{
  idx_t const nfactors = model->rank;
  val_t const * const restrict train_vals = train->vals;

  idx_t * perm = splatt_malloc(train->nnz * sizeof(*perm));

  val_t * predict_buffer = splatt_malloc(nfactors * sizeof(*predict_buffer));

  /* ensure lambda=1 */
  for(idx_t f=0; f < nfactors; ++f) {
    model->lambda[f] = 1.;
  }

  /* init perm */
  for(idx_t n=0; n < train->nnz; ++n) {
    perm[n] = n;
  }

  val_t prev_obj = 0;
  val_t prev_val_rmse = 0;

  /* foreach epoch */
  for(idx_t e=0; e < max_epochs; ++e) {
    /* new nnz ordering */
    shuffle_idx(perm, train->nnz);

    /* update model from all training observations */
    for(idx_t n=0; n < train->nnz; ++n) {
      p_update_model(train, perm[n], model, predict_buffer, learn_rate,
          regularization);
    }

    /* compute RMSE and adjust learning rate */
    val_t const obj = p_calc_obj(train, model, regularization);
    val_t const train_rmse = kruskal_rmse(train, model);
    val_t const val_rmse = kruskal_rmse(validate, model);
    printf("epoch:%4"SPLATT_PF_IDX"   obj: %0.5e   tr-rmse: %0.5e   v-rmse: %0.5e\n",
        e+1, obj, train_rmse, val_rmse);

    if(e > 0) {
      if(obj < prev_obj) {
        learn_rate *= 1.05;
      } else {
        learn_rate *= 0.50;
      }

      /* check convergence */
      if(fabs(val_rmse - prev_val_rmse) < 1e-8) {
        break;
      }
    }

    prev_obj = obj;
    prev_val_rmse = val_rmse;
  }

  splatt_free(predict_buffer);
  splatt_free(perm);
}


