
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "matrix.h"
#include "mttkrp.h"
#include "timer.h"
#include "thd_info.h"
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


static val_t __tt_kruskal_inner(
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
  MPI_Reduce(&myinner, &inner, 1, SS_MPI_VAL, MPI_SUM, 0, rinfo->comm_3d);
#else
  inner = myinner;
#endif

  return inner;
}


static val_t __calc_fit(
  idx_t const nmodes,
  rank_info * const rinfo,
  thd_info * const thds,
  idx_t const nthreads,
  val_t const ttnorm,
  val_t const * const restrict lambda,
  matrix_t ** mats,
  matrix_t const * const m1,
  matrix_t ** aTa)
{
  timer_start(&timers[TIMER_FIT]);

  /* First get norm of new model: lambda^T * (hada aTa) * lambda. */
  val_t const norm_mats = __kruskal_norm(nmodes, lambda, aTa);

  /* Compute inner product of tensor with new model */
  val_t const inner = __tt_kruskal_inner(nmodes, rinfo, thds, lambda, mats,m1);

  val_t const residual = sqrt(ttnorm + norm_mats - (2 * inner));
  timer_stop(&timers[TIMER_FIT]);
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
  ftensor_t ** ft,
  matrix_t ** mats,
  matrix_t ** globmats,
  val_t * const lambda,
  rank_info * const rinfo,
  cpd_opts const * const opts)
{
  idx_t const nfactors = opts->rank;
  idx_t const nmodes = ft[0]->nmodes;

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  omp_set_num_threads(opts->nthreads);
  thd_info * thds =  thd_init(opts->nthreads, 2,
    (nfactors * nfactors * sizeof(val_t)) + 64,
    TILE_SIZES[0] * nfactors * sizeof(val_t) + 64);

  matrix_t * m1 = mats[MAX_NMODES];

#ifdef SPLATT_USE_MPI
  /* Extract MPI communication structures */
  idx_t maxdim = 0;
  idx_t maxlocal2nbr = 0;
  idx_t maxnbr2globs = 0;
  for(idx_t m=0; m < nmodes; ++m) {
    maxlocal2nbr = SS_MAX(maxlocal2nbr, rinfo->nlocal2nbr[m]);
    maxnbr2globs = SS_MAX(maxnbr2globs, rinfo->nnbr2globs[m]);
    maxdim = SS_MAX(globmats[m]->I, maxdim);
  }
  maxlocal2nbr *= nfactors;
  maxnbr2globs *= nfactors;

  val_t * local2nbr_buf = (val_t *) malloc(maxlocal2nbr * sizeof(val_t));
  val_t * nbr2globs_buf = (val_t *) malloc(maxnbr2globs * sizeof(val_t));
  m1 = mat_alloc(maxdim, nfactors);

  /* Exchange initial matrices */
  for(idx_t m=1; m < nmodes; ++m) {
    mpi_update_rows(ft[m]->indmap, nbr2globs_buf, local2nbr_buf, mats[m],
        globmats[m], rinfo, nfactors, m);
  }
#endif

  /* Initialize first A^T * A mats. We redundantly do the first because it
   * makes communication easier. */
  matrix_t * aTa[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    aTa[m] = mat_alloc(nfactors, nfactors);
    mat_aTa(globmats[m], aTa[m], rinfo, thds, opts->nthreads);
  }
  /* used as buffer space */
  aTa[MAX_NMODES] = mat_alloc(nfactors, nfactors);

  /* Compute input tensor norm */
  val_t oldfit = 0;
  val_t mynorm = 0;
  #pragma omp parallel reduction(+:mynorm)
  for(idx_t n=0; n < ft[0]->nnz; ++n) {
    mynorm += ft[0]->vals[n] * ft[0]->vals[n];
  }
  val_t ttnormsq = 0;
#ifdef SPLATT_USE_MPI
  MPI_Allreduce(&mynorm, &ttnormsq, 1, SS_MPI_VAL, MPI_SUM, rinfo->comm_3d);
#else
  ttnormsq = mynorm;
#endif

  /* setup timers */
  timer_reset(&timers[TIMER_ATA]);
#ifdef SPLATT_USE_MPI
  timer_reset(&timers[TIMER_MPI]);
  timer_reset(&timers[TIMER_MPI_IDLE]);
  timer_reset(&timers[TIMER_MPI_COMM]);
  MPI_Barrier(rinfo->comm_3d);
#endif

  sp_timer_t itertime;
  timer_start(&timers[TIMER_CPD]);

  for(idx_t it=0; it < opts->niters; ++it) {
    timer_fstart(&itertime);
    for(idx_t m=0; m < nmodes; ++m) {
      mats[MAX_NMODES]->I = ft[0]->dims[m];
      m1->I = globmats[m]->I;

      /* M1 = X * (C o B) */
      timer_start(&timers[TIMER_MTTKRP]);
      mttkrp_splatt(ft[m], mats, m, thds, opts->nthreads);
      timer_stop(&timers[TIMER_MTTKRP]);
#ifdef SPLATT_USE_MPI
      /* TODO: do we actually need m1? */
      switch(opts->distribution) {
      case 1:
        memcpy(m1->vals, mats[MAX_NMODES]->vals, m1->I * nfactors * sizeof(val_t));
        break;
      default:
        /* add my partial multiplications to globmats[m] */
        mpi_add_my_partials(ft[m]->indmap, mats[MAX_NMODES], m1, rinfo,
            nfactors, m);
        /* incorporate neighbors' partials */
        mpi_reduce_rows(local2nbr_buf, nbr2globs_buf, mats[MAX_NMODES], m1,
            rinfo, nfactors, m);
        break;
      }
#endif

      /* M2 = (CtC .* BtB .* ...)^-1 */
      __calc_M2(m, nmodes, aTa);

      /* A = M1 * M2 */
      memset(globmats[m]->vals, 0, globmats[m]->I * nfactors * sizeof(val_t));
      mat_matmul(m1, aTa[MAX_NMODES], globmats[m]);

      /* normalize columns and extract lambda */
      if(it == 0) {
        mat_normalize(globmats[m], lambda, MAT_NORM_2, rinfo, thds,
            opts->nthreads);
      } else {
        mat_normalize(globmats[m], lambda, MAT_NORM_MAX, rinfo, thds,
            opts->nthreads);
      }

#ifdef SPLATT_USE_MPI
      /* send updated rows to neighbors */
      mpi_update_rows(ft[m]->indmap, nbr2globs_buf, local2nbr_buf, mats[m], globmats[m],
          rinfo, nfactors, m);
#endif

      /* update A^T*A */
      mat_aTa(globmats[m], aTa[m], rinfo, thds, opts->nthreads);
    } /* foreach mode */

    val_t const fit = __calc_fit(nmodes, rinfo, thds, opts->nthreads, ttnormsq,
        lambda, globmats, m1, aTa);
    timer_stop(&itertime);

    if(rinfo->rank == 0) {
      printf("    its = %3"SS_IDX" (%0.3fs)  fit = %0.5f  delta = %+0.5f\n",
          it+1, itertime.seconds, fit, fit - oldfit);
      oldfit = fit;
    }
  }
  timer_stop(&timers[TIMER_CPD]);

  /* clean up */
  for(idx_t m=0; m < nmodes; ++m) {
    mat_free(aTa[m]);
  }
  mat_free(aTa[MAX_NMODES]);

  thd_free(thds, opts->nthreads);
#ifdef SPLATT_USE_MPI
  mat_free(m1);
  free(local2nbr_buf);
  free(nbr2globs_buf);
#endif

#ifdef SPLATT_USE_MPI
  mpi_time_stats(rinfo);
#endif
}

