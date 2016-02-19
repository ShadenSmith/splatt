

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "complete.h"


#include <math.h>


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

val_t tc_rmse(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws)
{
  return sqrt(tc_loss_sq(test, model, ws) / test->nnz);
}





val_t tc_loss_sq(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws)
{
  val_t loss_obj = 0.;
  val_t const * const restrict train_vals = train->vals;

  #pragma omp parallel reduction(+:loss_obj)
  {
    val_t * buffer = ws->thds[omp_get_thread_num()].scratch[0];

    #pragma omp for schedule(static)
    for(idx_t x=0; x < train->nnz; ++x) {
      val_t const err = train_vals[x] - predict_val(model, train, x, buffer);
      loss_obj += err * err;
    }
  }

  return loss_obj;
}



val_t tc_frob_sq(
    tc_model const * const model,
    tc_ws const * const ws)
{
  assert(model->nmodes == test->nmodes);
  idx_t const nfactors = model->rank;

  val_t reg_obj = 0.;

  #pragma omp parallel reduction(+:reg_obj)
  {

    for(idx_t m=0; m < model->nmodes; ++m) {
      val_t accum = 0;
      val_t const * const restrict mat = model->factors[m];

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < model->dims[m] * nfactors; ++x) {
        accum += mat[x] * mat[x];
      }
      reg_obj += ws->regularization[m] * accum;
    }
  } /* end omp parallel */

  return reg_obj;
}



val_t tc_predict_val(
    tc_model const * const model,
    sptensor_t const * const tt,
    idx_t const index,
    val_t * const restrict buffer)
{


  return 0;
}


