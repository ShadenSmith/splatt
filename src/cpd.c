

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "kruskal.h"
#include "matrix.h"
#include "mttkrp.h"
#include "timer.h"
#include "thd_info.h"
#include "util.h"

#include <math.h>
#include <omp.h>

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


static val_t p_predict_val(
    val_t * const buffer)
{

}


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

int splatt_cpd_als(
    splatt_csf const * const tensors,
    splatt_idx_t const nfactors,
    double const * const options,
    splatt_kruskal * factored)
{
  matrix_t * mats[MAX_NMODES+1];

  idx_t nmodes = tensors->nmodes;

  rank_info rinfo;
  rinfo.rank = 0;

  /* allocate factor matrices */
  idx_t maxdim = tensors->dims[argmax_elem(tensors->dims, nmodes)];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (matrix_t *) mat_rand(tensors[0].dims[m], nfactors);
  }
  mats[MAX_NMODES] = mat_alloc(maxdim, nfactors);

  val_t * lambda = (val_t *) splatt_malloc(nfactors * sizeof(val_t));

  /* do the factorization! */
  factored->fit = cpd_als_iterate(tensors, mats, lambda, nfactors, &rinfo,
      options);

  /* store output */
  factored->rank = nfactors;
  factored->nmodes = nmodes;
  factored->lambda = lambda;
  for(idx_t m=0; m < nmodes; ++m) {
    factored->dims[m] = tensors->dims[m];
    factored->factors[m] = mats[m]->vals;
  }

  /* clean up */
  mat_free(mats[MAX_NMODES]);
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]); /* just the matrix_t ptr, data is safely in factored */
  }
  return SPLATT_SUCCESS;
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
double cpd_als_iterate(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  val_t * const lambda,
  idx_t const nfactors,
  rank_info * const rinfo,
  double const * const opts)
{
  idx_t const nmodes = tensors[0].nmodes;
  idx_t const nthreads = (idx_t) opts[SPLATT_OPTION_NTHREADS];

  /* Setup thread structures. + 64 bytes is to avoid false sharing.
   * TODO make this better */
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 3,
    (nfactors * nfactors * sizeof(val_t)) + 64,
    0,
    (nmodes * nfactors * sizeof(val_t)) + 64);

  matrix_t * m1 = mats[MAX_NMODES];

  /* Initialize first A^T * A mats. We redundantly do the first because it
   * makes communication easier. */
  matrix_t * aTa[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    aTa[m] = mat_alloc(nfactors, nfactors);
    mat_aTa(mats[m], aTa[m], rinfo, thds, nthreads);
  }
  /* used as buffer space */
  aTa[MAX_NMODES] = mat_alloc(nfactors, nfactors);

  /* Compute input tensor norm */
  double oldfit = 0;
  double fit = 0;
  val_t ttnormsq = csf_frobsq(tensors);

  /* setup timers */
  reset_cpd_timers();
  sp_timer_t itertime;
  sp_timer_t modetime[MAX_NMODES];
  timer_start(&timers[TIMER_CPD]);

  idx_t const niters = (idx_t) opts[SPLATT_OPTION_NITER];
  for(idx_t it=0; it < niters; ++it) {
    timer_fstart(&itertime);
    for(idx_t m=0; m < nmodes; ++m) {
      timer_fstart(&modetime[m]);
      mats[MAX_NMODES]->I = tensors[0].dims[m];
      m1->I = mats[m]->I;

      /* M1 = X * (C o B) */
      timer_start(&timers[TIMER_MTTKRP]);
      mttkrp_csf(tensors, mats, m, thds, opts);
      timer_stop(&timers[TIMER_MTTKRP]);

      /* M2 = (CtC .* BtB .* ...)^-1 */
      calc_gram_inv(m, nmodes, aTa);

      /* A = M1 * M2 */
      memset(mats[m]->vals, 0, mats[m]->I * nfactors * sizeof(val_t));
      mat_matmul(m1, aTa[MAX_NMODES], mats[m]);

      /* normalize columns and extract lambda */
      if(it == 0) {
        mat_normalize(mats[m], lambda, MAT_NORM_2, rinfo, thds, nthreads);
      } else {
        mat_normalize(mats[m], lambda, MAT_NORM_MAX, rinfo, thds,nthreads);
      }

      /* update A^T*A */
      mat_aTa(mats[m], aTa[m], rinfo, thds, nthreads);
      timer_stop(&modetime[m]);
    } /* foreach mode */

    fit = kruskal_calc_fit(nmodes, rinfo, thds, ttnormsq, lambda, mats, m1,
        aTa);
    timer_stop(&itertime);

    if(rinfo->rank == 0 &&
        opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_NONE) {
      printf("  its = %3"SPLATT_PF_IDX" (%0.3fs)  fit = %0.5f  delta = %+0.4e\n",
          it+1, itertime.seconds, fit, fit - oldfit);
      if(opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_LOW) {
        for(idx_t m=0; m < nmodes; ++m) {
          printf("     mode = %1"SPLATT_PF_IDX" (%0.3fs)\n", m+1,
              modetime[m].seconds);
        }
      }
    }
    if(it > 0 && fabs(fit - oldfit) < opts[SPLATT_OPTION_TOLERANCE]) {
      break;
    }
    oldfit = fit;
  }
  timer_stop(&timers[TIMER_CPD]);

  cpd_post_process(nfactors, nmodes, mats, lambda, thds, nthreads, rinfo);

  /* CLEAN UP */
  for(idx_t m=0; m < nmodes; ++m) {
    mat_free(aTa[m]);
  }
  mat_free(aTa[MAX_NMODES]);
  thd_free(thds, nthreads);

  return fit;
}



void cpd_post_process(
  idx_t const nfactors,
  idx_t const nmodes,
  matrix_t ** mats,
  val_t * const lambda,
  thd_info * const thds,
  idx_t const nthreads,
  rank_info * const rinfo)
{
  val_t * tmp =  splatt_malloc(nfactors * sizeof(*tmp));

  /* normalize each matrix and adjust lambda */
  for(idx_t m=0; m < nmodes; ++m) {
    mat_normalize(mats[m], tmp, MAT_NORM_2, rinfo, thds, nthreads);
    for(idx_t f=0; f < nfactors; ++f) {
      lambda[f] *= tmp[f];
    }
  }

  free(tmp);
}


