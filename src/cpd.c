
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
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static val_t __kruskal_norm(
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


/* Compute inner product of tensor with model
 * THIS OVERWRITES first row of aTa[MAX_NMODES] and mats[MAX_NMODES]*/
static val_t __tt_kruskal_inner(
  idx_t const nmodes,
  ftensor_t const * const ft,
  thd_info * const thds,
  idx_t const nthreads,
  val_t const * const restrict lambda,
  matrix_t ** mats,
  matrix_t ** aTa)
{
  idx_t const rank = mats[0]->J;
  val_t inner = 0;

  mats[MAX_NMODES]->I = mats[0]->I;
  mttkrp_splatt(ft, mats, 0, thds, nthreads);

  val_t const * const m0 = mats[0]->vals;
  val_t const * const mv = mats[MAX_NMODES]->vals;
  val_t       * const av = aTa[MAX_NMODES]->vals;

  for(idx_t r=0; r < rank; ++r) {
    av[r] = 0.;
  }
  for(idx_t i=0; i < ft->dims[0]; ++i) {
    for(idx_t r=0; r < rank; ++r) {
      av[r] += m0[r+(i*rank)] * mv[r+(i*rank)];
    }
  }
  /* accumulate everything into 'inner' */
  for(idx_t r=0; r < rank; ++r) {
    inner += av[r] * lambda[r];
  }
  return inner;
}


static val_t __calc_fit(
  idx_t const nmodes,
  ftensor_t const * const ft,
  thd_info * const thds,
  idx_t const nthreads,
  val_t const ttnorm,
  val_t const * const restrict lambda,
  matrix_t ** mats,
  matrix_t ** aTa)
{
  timer_start(&timers[TIMER_MISC]);

  /* First get norm of new model: lambda^T * (hada aTa) * lambda. */
  val_t const norm_mats = __kruskal_norm(nmodes, lambda, aTa);

  /* Compute inner product of tensor with new model */
  val_t const inner = __tt_kruskal_inner(nmodes, ft, thds, nthreads, lambda,
      mats, aTa);

  val_t const residual = sqrt(ttnorm + norm_mats - (2 * inner));
  timer_stop(&timers[TIMER_MISC]);
  return 1 - (residual / sqrt(ttnorm));
}


static void __calc_M2(
  idx_t const mode,
  idx_t const nmodes,
  matrix_t ** aTa)
{
  timer_start(&timers[TIMER_INV]);

  idx_t const rank = aTa[0]->J;
  val_t * const restrict av = aTa[MAX_NMODES]->vals;

  /* ata[MAX_NMODES] = hada(aTa[0], aTa[1], ...) */
  for(idx_t x=0; x < rank*rank; ++x) {
    av[x] = 1.;
  }
  for(idx_t m=1; m < nmodes; ++m) {
    idx_t const madjust = (mode + m) % nmodes;
    val_t const * const vals = aTa[madjust]->vals;
    for(idx_t x=0; x < rank*rank; ++x) {
      av[x] *= vals[x];
    }
  }

  /* M2 = M2^-1 */
  mat_syminv(aTa[MAX_NMODES]);
  timer_stop(&timers[TIMER_INV]);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void cpd(
  sptensor_t * const tt,
  matrix_t ** mats,
  cpd_opts const * const opts)
{
  idx_t const rank = opts->rank;
  idx_t const nmodes = tt->nmodes;

  /* allocate space for individual M^T * M matrices */
  matrix_t * aTa[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    aTa[m] = mat_alloc(rank, rank);
  }
  /* used as buffer space */
  aTa[MAX_NMODES] = mat_alloc(rank, rank);

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

  /* Initialize first A^T * A mats. We skip the first because it will be
   * solved for. */
  #pragma omp parallel for schedule(static, 1)
  for(idx_t m=1; m < nmodes; ++m) {
    mat_aTa(mats[m], aTa[m]);
  }

  val_t const ttnormsq = tt_normsq(tt);
  val_t const * const restrict xv = tt->vals;
  val_t oldfit = 0;

  timer_start(&timers[TIMER_CPD]);

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

      /* M2 = (CtC .* BtB .* ...)^-1 */
      __calc_M2(m, nmodes, aTa);

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

    val_t const fit = __calc_fit(nmodes, ft, thds, opts->nthreads, ttnormsq,
        lambda, mats, aTa);
    timer_stop(&itertime);

    printf("    its = %3"SS_IDX" (%0.3fs)  fit = %0.5f  delta = %+0.5f\n",
        it+1, itertime.seconds, fit, fit - oldfit);
    oldfit = fit;
  }
  timer_stop(&timers[TIMER_CPD]);

  for(idx_t m=0; m < nmodes; ++m) {
    mat_free(aTa[m]);
  }
  mat_free(aTa[MAX_NMODES]);

  ften_free(ft);
  thd_free(thds, opts->nthreads);
  free(lambda);
}

