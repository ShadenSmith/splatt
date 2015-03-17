
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "mttkrp.h"
#include "timer.h"
#include "thd_info.h"
#include "matrix.h"
#include "tile.h"
#include "io.h"

#include <math.h>
#include <omp.h>


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
void cpd_als(
    idx_t const rank,
    idx_t const nmodes,
    idx_t const nnz,
    idx_t ** const inds,
    val_t * const vals,
    val_t ** const mats,
    val_t * const lambda)
{

}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void cpd(
  sptensor_t * const tt,
  matrix_t ** mats,
  cpd_opts const * const opts)
{
  timer_start(&timers[TIMER_CPD]);
  idx_t const rank = opts->rank;
  idx_t const nmodes = tt->nmodes;

  /* allocate space for individual M^T * M matrices */
  matrix_t * aTa[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    aTa[m] = mat_alloc(rank, rank);
  }
  /* used as buffer space */
  aTa[MAX_NMODES] = mat_alloc(rank, rank);

  /* Initialize first A^T * A mats. We skip the first because it will be
   * solved for. */
  for(idx_t m=1; m < nmodes; ++m) {
    mat_aTa(mats[m], aTa[m]);
  }

  /* make accessing our buffers little easier */
  val_t * const mv = mats[MAX_NMODES]->vals;
  val_t * const av = aTa[MAX_NMODES]->vals;

  val_t * lambda = (val_t *) malloc(rank * sizeof(val_t));
  ftensor_t * ft = ften_alloc(tt, opts->tile);

  /* setup thread structures */
  omp_set_num_threads(opts->nthreads);
  thd_info * thds;
  if(opts->tile) {
    thds = thd_init(opts->nthreads, rank * sizeof(val_t) + 64,
      TILE_SIZES[0] * rank * sizeof(val_t) + 64);
  } else {
    thds = thd_init(opts->nthreads, rank * sizeof(val_t) + 64, 0);
  }

  val_t xnorm = 0.;
  val_t const * const restrict xv = tt->vals;
  #pragma omp parallel for reduction(+:xnorm)
  for(idx_t n=0; n < tt->nnz; ++n) {
    xnorm += xv[n] * xv[n];
  }

  val_t oldfit = 0;

  /* setup timers */
  sp_timer_t itertime;
  timer_reset(&timers[TIMER_SPLATT]);

  for(idx_t it=0; it < opts->niters; ++it) {
    timer_fstart(&itertime);
    for(idx_t m=0; m < nmodes; ++m) {

      /* M1 = X * (C o B) */
      mats[MAX_NMODES]->I = ft->dims[m];
      timer_start(&timers[TIMER_MTTKRP]);
      mttkrp_splatt(ft, mats, m, thds, opts->nthreads);
      timer_stop(&timers[TIMER_MTTKRP]);

#if 0
      if(it == 0) {
        char * fname = NULL;
        asprintf(&fname, "gold%"SS_IDX".mat", m);
        mat_write(mats[MAX_NMODES], fname);
        free(fname);
      }
      continue;
#endif
      /* M2 = (CtC .* BtB .* ...) */
      timer_start(&timers[TIMER_INV]);
      for(idx_t x=0; x < rank*rank; ++x) {
        av[x] = 1.;
      }
      for(idx_t mode=1; mode < nmodes; ++mode) {
        idx_t const madjust = (m + mode) % nmodes;
        val_t const * const atavals = aTa[madjust]->vals;
        for(idx_t x=0; x < rank*rank; ++x) {
          av[x] *= atavals[x];
        }
      }

      /* M2 = M2^-1 */
      mat_syminv(aTa[MAX_NMODES]);
      timer_stop(&timers[TIMER_INV]);

      /* A = M1 * M2 */
      memset(mats[m]->vals, 0, mats[m]->I * rank * sizeof(val_t));
      mat_matmul(mats[MAX_NMODES], aTa[MAX_NMODES], mats[m]);

      /* normalize columns and extract lambda */
      if(it == 0) {
        mat_normalize(mats[m], lambda, MAT_NORM_2);
      } else {
        mat_normalize(mats[m], lambda, MAT_NORM_MAX);
      }

      /* update A^T*A */
      mat_aTa(mats[m], aTa[m]);
    } /* foreach mode */


    /* CALCULATE FIT */

    /* First get fit of new model -- lambda^T * (hada aTa) * lambda. */
    /* calculate fit */
    val_t norm_mats = 0;
    for(idx_t x=0; x < rank*rank; ++x) {
      av[x] = 1.;
    }
    for(idx_t m=0; m < nmodes; ++m) {
      val_t const * const atavals = aTa[m]->vals;
      for(idx_t x=0; x < rank*rank; ++x) {
        av[x] *= atavals[x];
      }
    }
    for(idx_t i=0; i < rank; ++i) {
      for(idx_t j=0; j < rank; ++j) {
        norm_mats += av[j+(i*rank)] * lambda[i] * lambda[j];
      }
    }
    norm_mats = fabs(norm_mats);

    /* Compute inner product of tensor with model
     * THIS OVERWRITES first row of aTa[MAX_NMODES] and mats[MAX_NMODES]*/
    val_t inner = 0;
    for(idx_t r=0; r < rank; ++r) {
      mv[r] = 0.;
    }
    /* m-way product with each nnz */
    for(idx_t n=0; n < tt->nnz; ++n) {
      for(idx_t r=0; r < rank; ++r) {
        av[r] = tt->vals[n];
      }
      for(idx_t m=0; m < nmodes; ++m) {
        for(idx_t r=0; r < rank; ++r) {
          av[r] *= mats[m]->vals[r + (tt->ind[m][n] * rank)];
        }
      }
      for(idx_t r=0; r < rank; ++r) {
        mv[r] += av[r];
      }
    }
    /* accumulate everything into 'inner' */
    for(idx_t r=0; r < rank; ++r) {
      inner += mv[r] * lambda[r];
    }

    val_t residual = sqrt(xnorm + norm_mats - (2 * inner));
    val_t fit = 1 - (residual / sqrt(xnorm));

    timer_stop(&itertime);

    printf("    its = %3"SS_IDX" (%0.3fs)  fit = %0.5f  delta = %+0.5f\n",
        it+1, itertime.seconds, fit, fit - oldfit);
    oldfit = fit;
  }

  for(idx_t m=0; m < nmodes; ++m) {
    mat_free(aTa[m]);
  }
  mat_free(aTa[MAX_NMODES]);

  ften_free(ft);
  thd_free(thds, opts->nthreads);
  free(lambda);
  timer_stop(&timers[TIMER_CPD]);
}


