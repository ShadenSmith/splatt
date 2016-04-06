

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "completion.h"
#include "../io.h"
#include "../util.h"

#include <math.h>
#include <omp.h>



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Predict a value for a three-way tensor.
*
* @param model The model to use for the prediction.
* @param test The test tensor which gives us the model rows.
* @param index The nonzero to draw the row indices from. test[0][index] ...
*
* @return The predicted value.
*/
static inline val_t p_predict_val3(
    tc_model const * const model,
    sptensor_t const * const test,
    idx_t const index)
{
  val_t est = 0;
  idx_t const nfactors = model->rank;

  assert(test->nmodes == 3);

  idx_t const i = test->ind[0][index];
  idx_t const j = test->ind[1][index];
  idx_t const k = test->ind[2][index];

  val_t const * const restrict A = model->factors[0] + (i * nfactors);
  val_t const * const restrict B = model->factors[1] + (j * nfactors);
  val_t const * const restrict C = model->factors[2] + (k * nfactors);

  for(idx_t f=0; f < nfactors; ++f) {
    est += A[f] * B[f] * C[f];
  }

  return est;
}


/**
* @brief Predict a value for a three-way tensor when the model uses column-major
*        matrices.
*
* @param model The column-major model to use for the prediction.
* @param test The test tensor which gives us the model rows.
* @param index The nonzero to draw the row indices from. test[0][index] ...
*
* @return The predicted value.
*/
static inline val_t p_predict_val3_col(
    tc_model const * const model,
    sptensor_t const * const test,
    idx_t const index)
{
  val_t est = 0;
  idx_t const nfactors = model->rank;

  assert(test->nmodes == 3);

  idx_t const i = test->ind[0][index];
  idx_t const j = test->ind[1][index];
  idx_t const k = test->ind[2][index];

  idx_t const I = model->dims[0];
  idx_t const J = model->dims[1];
  idx_t const K = model->dims[2];

  val_t const * const restrict A = model->factors[0];
  val_t const * const restrict B = model->factors[1];
  val_t const * const restrict C = model->factors[2];

  for(idx_t f=0; f < nfactors; ++f) {
    est += A[i+(f*I)] * B[j+(f*J)] * C[k+(f*K)];
  }

  return est;
}


/**
* @brief Print some basic statistics about factorization progress.
*
* @param epoch Which epoch we are on.
* @param loss The sum-of-squared loss.
* @param rmse_tr The RMSE on the training set.
* @param rmse_vl The RMSE on the validation set.
* @param ws Workspace, used for timing information.
*/
static void p_print_progress(
    idx_t const epoch,
    val_t const loss,
    val_t const rmse_tr,
    val_t const rmse_vl,
    tc_ws const * const ws)
{
  printf("epoch:%4"SPLATT_PF_IDX"   loss: %0.5e   "
      "RMSE-tr: %0.5e   RMSE-vl: %0.5e   time: %0.3fs\n",
      epoch, loss, rmse_tr, rmse_vl, ws->tc_time.seconds);
}




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

val_t tc_rmse(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws)
{
#ifdef SPLATT_USE_MPI
  val_t loss = tc_loss_sq(test, model, ws);

  /* TODO: this portion is untimed, so recompute it each time for now */
  idx_t global_nnz = test->nnz;
  MPI_Allreduce(MPI_IN_PLACE, &global_nnz, 1, SPLATT_MPI_IDX, MPI_SUM,
      ws->rinfo->comm_3d);
  return sqrt(loss / global_nnz);
#else
  return sqrt(tc_loss_sq(test, model, ws) / test->nnz);
#endif
}



val_t tc_mae(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws)
{
  val_t loss_obj = 0.;
  val_t const * const restrict test_vals = test->vals;

  #pragma omp parallel reduction(+:loss_obj)
  {
    val_t * buffer = ws->thds[omp_get_thread_num()].scratch[0];

    if(model->which == SPLATT_TC_CCD) {
      #pragma omp for schedule(static)
      for(idx_t x=0; x < test->nnz; ++x) {
        val_t const predicted = tc_predict_val_col(model, test, x, buffer);
        loss_obj += fabs(test_vals[x] - predicted);
      }
    } else {
      #pragma omp for schedule(static)
      for(idx_t x=0; x < test->nnz; ++x) {
        val_t const predicted = tc_predict_val(model, test, x, buffer);
        loss_obj += fabs(test_vals[x] - predicted);
      }
    }
  }

#ifdef SPLATT_USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &loss_obj, 1, SPLATT_MPI_VAL, MPI_SUM,
      ws->rinfo->comm_3d);
  idx_t nnz = test->nnz;
  MPI_Allreduce(MPI_IN_PLACE, &nnz, 1, SPLATT_MPI_IDX, MPI_SUM,
      ws->rinfo->comm_3d);

  return loss_obj / nnz;
#else
  return loss_obj / test->nnz;
#endif
}




val_t tc_loss_sq(
    sptensor_t const * const test,
    tc_model const * const model,
    tc_ws * const ws)
{
  val_t loss_obj = 0.;
  val_t const * const restrict test_vals = test->vals;

  #pragma omp parallel reduction(+:loss_obj)
  {
    val_t * buffer = ws->thds[omp_get_thread_num()].scratch[0];

    if(model->which == SPLATT_TC_CCD) {
      #pragma omp for schedule(static)
      for(idx_t x=0; x < test->nnz; ++x) {
        val_t const err = test_vals[x] - tc_predict_val_col(model, test, x, buffer);
        loss_obj += err * err;
      }
    } else {
      #pragma omp for schedule(static)
      for(idx_t x=0; x < test->nnz; ++x) {
        val_t const err = test_vals[x] - tc_predict_val(model, test, x, buffer);
        loss_obj += err * err;
      }
    }
  }

#ifdef SPLATT_USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &loss_obj, 1, SPLATT_MPI_VAL, MPI_SUM,
      ws->rinfo->comm_3d);
#endif

  return loss_obj;
}



val_t tc_frob_sq(
    tc_model const * const model,
    tc_ws const * const ws)
{
  idx_t const nfactors = model->rank;

  val_t reg_obj = 0.;

  #pragma omp parallel reduction(+:reg_obj)
  {
    for(idx_t m=0; m < model->nmodes; ++m) {
      val_t accum = 0;
#ifdef SPLATT_USE_MPI
      idx_t const nrows = model->globmats[m]->I;
      val_t const * const restrict mat = model->globmats[m]->vals;
#else
      idx_t const nrows = model->dims[m];
      val_t const * const restrict mat = model->factors[m];
#endif

      #pragma omp for schedule(static) nowait
      for(idx_t x=0; x < nrows * nfactors; ++x) {
        accum += mat[x] * mat[x];
      }
      reg_obj += ws->regularization[m] * accum;
    }
  } /* end omp parallel */

#ifdef SPLATT_USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &reg_obj, 1, SPLATT_MPI_VAL, MPI_SUM,
      ws->rinfo->comm_3d);
#endif

  assert(reg_obj > 0);
  return reg_obj;
}



val_t tc_predict_val(
    tc_model const * const model,
    sptensor_t const * const test,
    idx_t const index,
    val_t * const restrict buffer)
{
  if(test->nmodes == 3) {
    return p_predict_val3(model, test, index);
  }

  idx_t const nfactors = model->rank;

  /* initialize accumulation of each latent factor with the first row */
  idx_t const row_id = test->ind[0][index];
  val_t const * const init_row = model->factors[0] + (row_id * nfactors);
  for(idx_t f=0; f < nfactors; ++f) {
    buffer[f] = init_row[f];
  }

  /* now multiply each factor by A(i,:), B(j,:) ... */
  idx_t const nmodes = model->nmodes;
  for(idx_t m=1; m < nmodes; ++m) {
    idx_t const row_id = test->ind[m][index];
    val_t const * const row = model->factors[m] + (row_id * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buffer[f] *= row[f];
    }
  }

  /* finally, sum the factors to form the final estimated value */
  val_t est = 0;
  for(idx_t f=0; f < nfactors; ++f) {
    est += buffer[f];
  }

  return est;
}


val_t tc_predict_val_col(
    tc_model const * const model,
    sptensor_t const * const test,
    idx_t const index,
    val_t * const restrict buffer)
{
  if(test->nmodes == 3) {
    return p_predict_val3_col(model, test, index);
  }

  idx_t const nfactors = model->rank;

  /* initialize accumulation of each latent factor with the first row */
  idx_t const row_id = test->ind[0][index];
  val_t const * const init_row = model->factors[0] + (row_id * nfactors);
  for(idx_t f=0; f < nfactors; ++f) {
    buffer[f] = model->factors[0][row_id + (f * model->dims[0])];
  }

  /* now multiply each factor by A(i,:), B(j,:) ... */
  idx_t const nmodes = model->nmodes;
  for(idx_t m=1; m < nmodes; ++m) {
    idx_t const row_id = test->ind[m][index];
    for(idx_t f=0; f < nfactors; ++f) {
      buffer[f] *= model->factors[m][row_id + (f * model->dims[m])];
    }
  }

  /* finally, sum the factors to form the final estimated value */
  val_t est = 0;
  for(idx_t f=0; f < nfactors; ++f) {
    est += buffer[f];
  }

  return est;
}


/******************************************************************************
 * WORKSPACE FUNCTIONS
 *****************************************************************************/

tc_model * tc_model_alloc(
    sptensor_t const * const train,
    idx_t const rank,
    splatt_tc_type const which)
{
  tc_model * model = splatt_malloc(sizeof(*model));

  model->which = which;
  model->rank = rank;
  model->nmodes = train->nmodes;
  for(idx_t m=0; m < train->nmodes; ++m) {
    model->dims[m] = train->dims[m];

    size_t const bytes = model->dims[m] * rank * sizeof(**(model->factors));
    model->factors[m] = splatt_malloc(bytes);
    fill_rand(model->factors[m], model->dims[m] * rank);
  }

  return model;
}


tc_model * tc_model_copy(
    tc_model const * const model)
{
  tc_model * ret = splatt_malloc(sizeof(*model));

  ret->which = model->which;
  ret->rank = model->rank;
  ret->nmodes = model->nmodes;
  for(idx_t m=0; m < model->nmodes; ++m) {
    ret->dims[m] = model->dims[m];

    idx_t const bytes = model->dims[m] * model->rank *
        sizeof(**(model->factors));
    ret->factors[m] = splatt_malloc(bytes);
    par_memcpy(ret->factors[m], model->factors[m], bytes);
  }

  return ret;
}


void tc_model_free(
    tc_model * model)
{
  for(idx_t m=0; m < model->nmodes; ++m) {
    splatt_free(model->factors[m]);

#ifdef SPLATT_USE_MPI
    mat_free(model->globmats[m]);
#endif
  }

  splatt_free(model);
}


tc_ws * tc_ws_alloc(
    sptensor_t const * const train,
    tc_model const * const model,
    idx_t nthreads)
{
  tc_ws * ws = splatt_malloc(sizeof(*ws));

  idx_t const nmodes = model->nmodes;
  ws->nmodes = nmodes;

  /* some reasonable defaults */
  ws->learn_rate = 0.001;
  idx_t const rank = model->rank;

  ws->maxdense_dim = 0;
  ws->num_dense =0;

  /* Set mode-specific parameters, etc. */
  for(idx_t m=0; m < nmodes; ++m) {
    /* dense modes */
    ws->isdense[m] = train->dims[m] < DENSEMODE_THRESHOLD;
    if(ws->isdense[m]) {
      ws->maxdense_dim = SS_MAX(ws->maxdense_dim, train->dims[m]);
      ++(ws->num_dense);
    }

    switch(model->which) {
    case SPLATT_TC_GD:
      ws->regularization[m] = 1e-2;
      break;
    case SPLATT_TC_NLCG:
      ws->regularization[m] = 1e-2;
      break;
    case SPLATT_TC_LBFGS:
      ws->regularization[m] = 1e-2;
      break;
    case SPLATT_TC_SGD:
      ws->regularization[m] = 5e-3;
      break;
    case SPLATT_TC_CCD:
      ws->regularization[m] = 1e2;
      break;
    case SPLATT_TC_ALS:
      ws->regularization[m] = 1e2;
      break;
    case SPLATT_TC_NALGS:
      break;
    }
  }

  /* reset timers */
  timer_reset(&ws->grad_time);
  timer_reset(&ws->line_time);
  timer_reset(&ws->tc_time);

  /* size of largest mode (for CCD) */
  idx_t const max_dim = train->dims[argmax_elem(train->dims, train->nmodes)];

  /* Allocate general structures */
  ws->numerator = NULL;
  ws->denominator = NULL;
  ws->nthreads = nthreads;
  switch(model->which) {
  case SPLATT_TC_GD:
    ws->thds = thd_init(nthreads, 1, rank * sizeof(val_t));
    break;
  case SPLATT_TC_LBFGS:
    ws->thds = thd_init(nthreads, 1, rank * sizeof(val_t));
    break;
  case SPLATT_TC_NLCG:
    ws->thds = thd_init(nthreads, 1, rank * sizeof(val_t));
    break;
  case SPLATT_TC_SGD:
    ws->thds = thd_init(nthreads, 1, rank * sizeof(val_t));
    break;
  case SPLATT_TC_CCD:
    ws->numerator   = splatt_malloc(max_dim * sizeof(*ws->numerator));
    ws->denominator = splatt_malloc(max_dim * sizeof(*ws->denominator));
    ws->thds = thd_init(nthreads, 1, rank * sizeof(val_t));
    break;
  case SPLATT_TC_ALS:
    ws->thds = thd_init(nthreads, 4,
        rank * sizeof(val_t),                /* prediction buffer */
        rank * sizeof(val_t),                /* MTTKRP buffer */
        rank * rank * sizeof(val_t),         /* normal equations */
        rank * ALS_BUFSIZE * sizeof(val_t)); /* pre-normal equations buffer */
    break;
  case SPLATT_TC_NALGS:
    ws->thds = NULL;
    fprintf(stderr, "SPLATT: completion algorithm not recognized.\n");
    break;
  }

  /* convergence */
  ws->max_its = 1000;
  ws->num_inner = 1;
  ws->max_seconds = 1000;
  ws->max_badepochs = 20;
  ws->nbadepochs = 0;
  ws->best_epoch = 0;
  ws->best_rmse = SPLATT_VAL_MAX;
  ws->tolerance = 1e-4;

  ws->best_model = tc_model_copy(model);

#ifdef SPLATT_USE_MPI
  ws->nbr2globs_buf  = NULL;
  ws->local2nbr_buf  = NULL;
#endif

  return ws;
}


void tc_ws_free(
    tc_ws * ws)
{
  thd_free(ws->thds, ws->nthreads);
  splatt_free(ws->numerator);
  splatt_free(ws->denominator);

#ifdef SPLATT_USE_MPI
  splatt_free(ws->nbr2globs_buf);
  splatt_free(ws->local2nbr_buf);
  rank_free(*(ws->rinfo), ws->best_model->nmodes);
#endif

  tc_model_free(ws->best_model);

  splatt_free(ws);
}


bool tc_converge(
    sptensor_t const * const train,
    sptensor_t const * const validate,
    tc_model const * const model,
    val_t const loss,
    val_t const frobsq,
    idx_t const epoch,
    tc_ws * const ws)
{
#ifdef SPLATT_USE_MPI
  val_t const train_rmse = sqrt(loss / ws->rinfo->global_nnz);
#else
  val_t const train_rmse = sqrt(loss / train->nnz);
#endif

  timer_stop(&ws->tc_time);
  val_t const val_rmse = tc_rmse(validate, model, ws);
#ifdef SPLATT_USE_MPI
  if(ws->rinfo->rank == 0) {
    p_print_progress(epoch, loss, train_rmse, val_rmse, ws);
  }
#else
  p_print_progress(epoch, loss, train_rmse, val_rmse, ws);
#endif

  bool converged = false;
  if(val_rmse - ws->best_rmse < -(ws->tolerance)) {
    ws->nbadepochs = 0;
    ws->best_rmse = val_rmse;
    ws->best_epoch = epoch;

    /* save the best model */
    for(idx_t m=0; m < model->nmodes; ++m) {
      par_memcpy(ws->best_model->factors[m], model->factors[m],
          model->dims[m] * model->rank * sizeof(**(model->factors)));
      /* TODO copy globmats too */
    }
  } else {
    ++ws->nbadepochs;
    if(ws->nbadepochs == ws->max_badepochs) {
      converged = true;
    }
  }

  /* check for time limit */
  if(ws->max_seconds > 0 && ws->tc_time.seconds >= ws->max_seconds) {
    converged = true;
  }

  if(!converged) {
    timer_start(&ws->tc_time);
  }
  return converged;
}




/******************************************************************************
 * MPI FUNCTIONS
 *****************************************************************************/

#ifdef SPLATT_USE_MPI

int mpi_tc_distribute_coarse(
    char const * const train_fname,
    char const * const validate_fname,
    idx_t const * const dims,
    sptensor_t * * train_out,
    sptensor_t * * validate_out,
    rank_info * const rinfo)
{
  rinfo->decomp = SPLATT_DECOMP_COARSE;
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    rinfo->dims_3d[m] = 1;
  }

  /* distribute training tensor with a coarse-grained decomposition */
  sptensor_t * train = mpi_tt_read(train_fname, NULL, rinfo);
  if(train == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }
  *train_out = train;

  /* simple distribution for validate tensor...
   * TODO: redistribute to match training distribution */
  sptensor_t * val_tmp = mpi_simple_distribute(validate_fname, MPI_COMM_WORLD);
  if(val_tmp == NULL) {
    tt_free(train);
    *train_out = NULL;
    return SPLATT_ERROR_BADINPUT;
  }

  idx_t const bigmode = argmax_elem(train->dims, train->nmodes);
  idx_t * coarse_parts = calloc(rinfo->npes+1, sizeof(*coarse_parts));
  coarse_parts[rinfo->rank] = rinfo->mat_start[bigmode];
  MPI_Allreduce(MPI_IN_PLACE, coarse_parts, rinfo->npes, SPLATT_MPI_IDX,
      MPI_SUM, rinfo->comm_3d);
  coarse_parts[rinfo->npes] = train->dims[bigmode];

  /* redistribute along largest mode */
  int * parts = splatt_malloc(val_tmp->nnz * sizeof(*parts));
  idx_t const * const bigind = val_tmp->ind[bigmode];
  #pragma omp parallel for schedule(static)
  for(idx_t n=0; n < val_tmp->nnz; ++n) {
    for(int p=0; p < rinfo->npes; ++p) {
      if(coarse_parts[p+1] > bigind[n]) {
        parts[n] = p;
        break;
      }
    }
  }
  free(coarse_parts);

  /* rearrange validation nonzeros */
  sptensor_t * validate = mpi_rearrange_by_part(val_tmp, parts,rinfo->comm_3d);
  *validate_out = validate;
  tt_free(val_tmp);
  splatt_free(parts);

  return SPLATT_SUCCESS;
}



int mpi_tc_distribute_med(
    char const * const train_fname,
    char const * const validate_fname,
    idx_t const * const dims,
    sptensor_t * * train_out,
    sptensor_t * * validate_out,
    rank_info * const rinfo)
{
  /* defaults if not otherwise specified */
  if(dims == NULL) {
    rinfo->decomp = DEFAULT_MPI_DISTRIBUTION;
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      rinfo->dims_3d[m] = 1;
    }
  } else {
    rinfo->decomp = SPLATT_DECOMP_MEDIUM;
    memcpy(rinfo->dims_3d, dims, MAX_NMODES * sizeof(*(rinfo->dims_3d)));
  }

  /* distribute training tensor with a medium-grained decomposition */
  sptensor_t * train = mpi_tt_read(train_fname, NULL, rinfo);
  *train_out = train;
  if(train == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  /* simple distribution for validate tensor...we will redistribute to match
   * training distribution */
  sptensor_t * val_tmp = mpi_simple_distribute(validate_fname, rinfo->comm_3d);
  if(val_tmp == NULL) {
    tt_free(train);
    *train_out = NULL;
    return SPLATT_ERROR_BADINPUT;
  }

  int * parts = splatt_malloc(val_tmp->nnz * sizeof(*parts));
  #pragma omp parallel for schedule(static)
  for(idx_t n=0; n < val_tmp->nnz; ++n) {
    parts[n] = mpi_determine_med_owner(val_tmp, n, rinfo);
  }

  /* rearrange validation nonzeros */
  sptensor_t * validate = mpi_rearrange_by_part(val_tmp, parts,rinfo->comm_3d);
  *validate_out = validate;
  tt_free(val_tmp);
  splatt_free(parts);

  /* now map validation indices to layer coordinates and fill in dims */
  #pragma omp parallel
  for(idx_t m=0; m < validate->nmodes; ++m) {
    #pragma omp master
    validate->dims[m] = rinfo->layer_ends[m] - rinfo->layer_starts[m];

    #pragma omp for schedule(static) nowait
    for(idx_t n=0; n < validate->nnz; ++n) {
      assert(validate->ind[m][n] >= rinfo->layer_starts[m]);
      assert(validate->ind[m][n] < rinfo->layer_ends[m]);
      validate->ind[m][n] -= rinfo->layer_starts[m];
    }
  }

  return SPLATT_SUCCESS;
}


tc_model * mpi_tc_model_alloc(
    sptensor_t const * const train,
    idx_t const rank,
    splatt_tc_type const which,
    permutation_t const * const perm,
    rank_info * const rinfo)
{
  tc_model * model = splatt_malloc(sizeof(*model));

  model->which = which;
  model->rank = rank;
  model->nmodes = train->nmodes;
  for(idx_t m=0; m < train->nmodes; ++m) {
    /* just allocate local factors */
    model->dims[m] = train->dims[m];
    model->factors[m] = splatt_malloc(model->dims[m] * rank *
        sizeof(**model->factors));

    /* use mpi_mat_rand() to get consistent initializations across runs with
     * different numbers of MPI ranks */
    model->globmats[m] = mpi_mat_rand(m, rank, perm, rinfo);
  }

  return model;
}




#endif








