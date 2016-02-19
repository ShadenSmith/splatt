
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "kruskal.h"

#include <math.h>
#include <omp.h>


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

val_t predict_val(
    splatt_kruskal const * const factored,
    sptensor_t const * const tt,
    idx_t const index,
    val_t * const restrict accum)
{
  /* initialize accumulation of each latent factor with lambda(r) */
  idx_t const nfactors = factored->rank;
  for(idx_t f=0; f < nfactors; ++f) {
    accum[f] = factored->lambda[f];
  }

  /* now multiply each factor by A(i,:), B(j,:) ... */
  idx_t const nmodes = factored->nmodes;
  for(idx_t m=0; m < nmodes; ++m) {
    idx_t const row_id = tt->ind[m][index];
    val_t const * const row = factored->factors[m] + (row_id * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      accum[f] *= row[f];
    }
  }

  /* finally, sum the factors to form the final estimated value */
  val_t est = 0;
  for(idx_t f=0; f < nfactors; ++f) {
    est += accum[f];
  }
  return est;
}



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

void splatt_free_kruskal(
    splatt_kruskal * factored)
{
  free(factored->lambda);
  for(idx_t m=0; m < factored->nmodes; ++m) {
    free(factored->factors[m]);
  }
}


int splatt_kruskal_predict(
    splatt_kruskal const * const factored,
    splatt_idx_t const * const coords,
    splatt_val_t * const predicted)
{
  /* check for out of bounds */
  for(idx_t m=0; m < factored->nmodes; ++m) {
    if(coords[m] >= factored->dims[m]) {
      return SPLATT_ERROR_BADINPUT;
    }
  }

  /* initialize accumulation of each latent factor with lambda(r) */
  idx_t const nfactors = factored->rank;
  val_t * restrict accum = splatt_malloc(nfactors * sizeof(*accum));
  for(idx_t f=0; f < nfactors; ++f) {
    accum[f] = factored->lambda[f];
  }

  /* now multiply each factor by A(i,:), B(j,:) ... */
  for(idx_t m=0; m < factored->nmodes; ++m) {
    val_t const * const restrict row = factored->factors[m] +
        (coords[m] * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      accum[f] *= row[f];
    }
  }

  /* finally, sum the factors to form the final estimated value */
  val_t est = 0;
  for(idx_t f=0; f < nfactors; ++f) {
    est += accum[f];
  }

  splatt_free(accum);

  *predicted = est;
  return SPLATT_SUCCESS;
}




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

val_t kruskal_calc_fit(
  idx_t const nmodes,
  rank_info * const rinfo,
  thd_info * const thds,
  val_t const ttnormsq,
  val_t const * const restrict lambda,
  matrix_t ** mats,
  matrix_t const * const mttkrp,
  matrix_t ** aTa)
{
  timer_start(&timers[TIMER_FIT]);

  /* First get norm of new model: lambda^T * (hada aTa) * lambda. */
  val_t const norm_mats = kruskal_norm(nmodes, lambda, aTa);

  /* Compute inner product of tensor with new model */
  val_t const inner = kruskal_mttkrp_inner(nmodes, rinfo, thds, lambda, mats,
      mttkrp);

  val_t const residual = sqrt(ttnormsq + norm_mats - (2 * inner));
  timer_stop(&timers[TIMER_FIT]);
  return 1 - (residual / sqrt(ttnormsq));
}

val_t kruskal_mttkrp_inner(
  idx_t const nmodes,
  rank_info * const rinfo,
  thd_info * const thds,
  val_t const * const restrict lambda,
  matrix_t ** mats,
  matrix_t const * const m1)
{
  idx_t const rank = mats[0]->J;
  idx_t const lastm = nmodes - 1;
  idx_t const dim = m1->I;

  val_t const * const m0 = mats[lastm]->vals;
  val_t const * const mv = m1->vals;

  val_t myinner = 0;
  #pragma omp parallel reduction(+:myinner)
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];

    for(idx_t r=0; r < rank; ++r) {
      accumF[r] = 0.;
    }

    #pragma omp for
    for(idx_t i=0; i < dim; ++i) {
      for(idx_t r=0; r < rank; ++r) {
        accumF[r] += m0[r+(i*rank)] * mv[r+(i*rank)];
      }
    }
    /* accumulate everything into 'myinner' */
    for(idx_t r=0; r < rank; ++r) {
      myinner += accumF[r] * lambda[r];
    }
  }
  val_t inner = 0.;

#ifdef SPLATT_USE_MPI
  timer_start(&timers[TIMER_MPI_FIT]);
  timer_start(&timers[TIMER_MPI_IDLE]);
  MPI_Barrier(rinfo->comm_3d);
  timer_stop(&timers[TIMER_MPI_IDLE]);

  MPI_Allreduce(&myinner, &inner, 1, SPLATT_MPI_VAL, MPI_SUM, rinfo->comm_3d);
  timer_stop(&timers[TIMER_MPI_FIT]);
#else
  inner = myinner;
#endif

  return inner;
}



val_t kruskal_norm(
  idx_t const nmodes,
  val_t const * const restrict lambda,
  matrix_t ** aTa)
{
  idx_t const rank = aTa[0]->J;
  val_t * const restrict av = aTa[MAX_NMODES]->vals;

  val_t norm_mats = 0;

  /* use aTa[MAX_NMODES] as scratch space */
  for(idx_t x=0; x < rank*rank; ++x) {
    av[x] = 1.;
  }

  /* aTa[MAX_NMODES] = hada(aTa) */
  for(idx_t m=0; m < nmodes; ++m) {
    val_t const * const restrict atavals = aTa[m]->vals;
    for(idx_t x=0; x < rank*rank; ++x) {
      av[x] *= atavals[x];
    }
  }

  /* now compute lambda^T * aTa[MAX_NMODES] * lambda */
  for(idx_t i=0; i < rank; ++i) {
    for(idx_t j=0; j < rank; ++j) {
      norm_mats += av[j+(i*rank)] * lambda[i] * lambda[j];
    }
  }

  return fabs(norm_mats);
}


val_t kruskal_loss(
    sptensor_t const * const test,
    splatt_kruskal const * const factored)
{
  val_t loss = 0;
  #pragma omp parallel reduction(+:loss)
  {
    val_t * loss_buffer = splatt_malloc(factored->rank * sizeof(*loss_buffer));
    val_t const * const restrict vals = test->vals;

    #pragma omp for schedule(static)
    for(idx_t x=0; x < test->nnz; ++x) {
      val_t const est = predict_val(factored, test, x, loss_buffer);
      loss += (vals[x] - est) * (vals[x] - est);
    }
    splatt_free(loss_buffer);
  }

  return loss;
}


val_t kruskal_rmse(
    sptensor_t const * const test,
    splatt_kruskal const * const factored)
{
  return sqrt(kruskal_loss(test, factored) / test->nnz);
}



