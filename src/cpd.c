
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

  val_t * lambda = (val_t *) malloc(rank * sizeof(val_t));
  matrix_t * ata    = mat_alloc(rank, rank);
  matrix_t * atabuf = mat_alloc(rank, rank);
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

  /* setup timers */
  sp_timer_t itertime;
  timer_reset(&timers[TIMER_SPLATT]);

  for(idx_t it=0; it < opts->niters; ++it) {
    timer_fstart(&itertime);
    for(idx_t m=0; m < nmodes; ++m) {
      mats[MAX_NMODES]->I = ft->dims[m];

      /* M1 = X * (C o B) */
      timer_start(&timers[TIMER_MTTKRP]);
      mttkrp_splatt(ft, mats, m, thds, opts->nthreads);
      timer_stop(&timers[TIMER_MTTKRP]);

#if 1
      if(it == 0) {
        char * fname = NULL;
        asprintf(&fname, "gold%"SS_IDX".mat", m);
        mat_write(mats[MAX_NMODES], fname);
        free(fname);
      }
#endif
      continue;

      timer_start(&timers[TIMER_INV]);
      /* M2 = (CtC * BtB * ...) */
      mat_aTa_hada(mats, (m+1) % nmodes, nmodes-1, nmodes, atabuf, ata);

      /* M2 = M2^-1 */
      mat_syminv(ata);
      timer_stop(&timers[TIMER_INV]);

      /* A = M1 * M2 */
      memset(mats[m]->vals, 0, mats[m]->I * rank * sizeof(val_t));
      mat_matmul(mats[MAX_NMODES], ata, mats[m]);

      /* normalize columns and extract lambda if necessary */
      if(it == 0) {
        mat_normalize(mats[m], lambda, MAT_NORM_2);
      } else {
        mat_normalize(mats[m], lambda, MAT_NORM_MAX);
      }
    }

#if 0
    /* calculate fit */
    val_t norm_mats = 0;
    mat_aTa_hada(mats, 0, nmodes, nmodes, atabuf, ata);
    /* add lambda * lambda^T to ata */
    for(idx_t i=0; i < rank; ++i) {
      for(idx_t j=0; j < rank; ++j) {
        norm_mats += ata->vals[j+(i*rank)] * lambda[i] * lambda[j];
      }
    }
    norm_mats = fabs(norm_mats);

    /* compute inner product */
    val_t inner = 0;
    val_t * const mv = mats[MAX_NMODES]->vals;
    val_t * const av = atabuf->vals;
    for(idx_t r=0; r < rank; ++r) {
      mv[r] = 0.;
    }
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
    for(idx_t r=0; r < rank; ++r) {
      inner += mv[r] * lambda[r];
    }

    val_t residual = sqrt(xnorm + norm_mats - (2 * inner));
    val_t fit = 1 - (residual / sqrt(xnorm));
#endif
    val_t fit = 0.1;

    timer_stop(&itertime);
    printf("    its = %3"SS_IDX" (%0.3fs)  fit = %0.3f\n", it+1,
        itertime.seconds, fit);
  }

  ften_free(ft);
  thd_free(thds, opts->nthreads);
  mat_free(ata);
  mat_free(atabuf);
  free(lambda);
  timer_stop(&timers[TIMER_CPD]);
}


