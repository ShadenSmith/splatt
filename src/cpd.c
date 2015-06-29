
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "matrix.h"
#include "mttkrp.h"
#include "sptensor.h"
#include "stats.h"
#include "timer.h"
#include "thd_info.h"
#include "tile.h"
#include "io.h"
#include "util.h"

#include <math.h>
#include <omp.h>



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
int splatt_cpd(
    splatt_idx_t const nfactors,
    splatt_idx_t const nmodes,
    splatt_csf_t ** tensors,
    splatt_val_t ** const mats,
    splatt_val_t * const lambda,
    double const * const options)
{
  matrix_t * globmats[MAX_NMODES+1];

  rank_info rinfo;
  rinfo.rank = 0;

  idx_t maxdim = 0;
  for(idx_t m=0; m < nmodes; ++m) {
    globmats[m] = (matrix_t *) malloc(sizeof(matrix_t));
    globmats[m]->I = tensors[0]->dims[m];
    globmats[m]->J = nfactors;
    globmats[m]->vals = mats[m];
    globmats[m]->rowmajor = 1;

    fill_rand(globmats[m]->vals, globmats[m]->I * globmats[m]->J);
    maxdim = SS_MAX(globmats[m]->I, maxdim);
  }
  globmats[MAX_NMODES] = mat_alloc(maxdim, nfactors);

  /* do the factorization! */
  cpd_als(tensors, globmats, globmats, lambda, nfactors, &rinfo, options);

  mat_free(globmats[MAX_NMODES]);
  for(idx_t m=0; m < nmodes; ++m) {
    free(globmats[m]); /* just the matrix_t ptr, data is safely in mats */
  }
  return SPLATT_SUCCESS;
}


#if 0
int splatt_cpd(
    idx_t const nfactors,
    idx_t const nmodes,
    idx_t const nnz,
    idx_t ** const inds,
    val_t * const vals,
    val_t ** const mats,
    val_t * const lambda,
    double const * const options)
{
  sptensor_t tt;
  tt_fill(&tt, nnz, nmodes, inds, vals);
  tt_remove_empty(&tt);

  matrix_t * globmats[MAX_NMODES+1];

  rank_info rinfo;
  rinfo.rank = 0;

  /* fill each ftensor */
  idx_t maxdim = 0;
  ftensor_t * ft[MAX_NMODES];
  for(idx_t m=0; m < tt.nmodes; ++m) {
    ft[m] = ften_alloc(&tt, m, (int) options[SPLATT_OPTION_TILE]);

    globmats[m] = (matrix_t *) malloc(sizeof(matrix_t));
    globmats[m]->I = tt.dims[m];
    globmats[m]->J = nfactors;
    globmats[m]->vals = mats[m];
    globmats[m]->rowmajor = 1;
    fill_rand(globmats[m]->vals, globmats[m]->I * globmats[m]->J);

    maxdim = SS_MAX(globmats[m]->I, maxdim);
  }
  globmats[MAX_NMODES] = mat_alloc(maxdim, nfactors);

  /* do the factorization! */
  cpd_als(ft, globmats, globmats, lambda, nfactors, &rinfo, options);

  mat_free(globmats[MAX_NMODES]);
  for(idx_t m=0; m < tt.nmodes; ++m) {
    ften_free(ft[m]);
    free(globmats[m]);
  }
  return SPLATT_SUCCESS;
}
#endif


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Resets serial and MPI timers that were activated during some CPD
*        pre-processing.
*
* @param rinfo MPI rank information.
*/
static void __reset_cpd_timers(
  rank_info const * const rinfo)
{
  timer_reset(&timers[TIMER_ATA]);
#ifdef SPLATT_USE_MPI
  timer_reset(&timers[TIMER_MPI]);
  timer_reset(&timers[TIMER_MPI_IDLE]);
  timer_reset(&timers[TIMER_MPI_COMM]);
  timer_reset(&timers[TIMER_MPI_ATA]);
  timer_reset(&timers[TIMER_MPI_REDUCE]);
  timer_reset(&timers[TIMER_MPI_NORM]);
  timer_reset(&timers[TIMER_MPI_UPDATE]);
  timer_reset(&timers[TIMER_MPI_FIT]);
  MPI_Barrier(rinfo->comm_3d);
#endif
}


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
  timer_start(&timers[TIMER_MPI_FIT]);
  timer_start(&timers[TIMER_MPI_IDLE]);
  MPI_Barrier(rinfo->comm_3d);
  timer_stop(&timers[TIMER_MPI_IDLE]);

  MPI_Allreduce(&myinner, &inner, 1, SS_MPI_VAL, MPI_SUM, rinfo->comm_3d);
  timer_stop(&timers[TIMER_MPI_FIT]);
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
void cpd_als(
  ftensor_t ** ft,
  matrix_t ** mats,
  matrix_t ** globmats,
  val_t * const lambda,
  idx_t const nfactors,
  rank_info * const rinfo,
  double const * const opts)
{
  idx_t const nmodes = ft[0]->nmodes;
  idx_t const nthreads = (idx_t) opts[SPLATT_OPTION_NTHREADS];

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 2,
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
  if(rinfo->distribution == 3) {
    m1 = mat_alloc(maxdim, nfactors);
  }

  /* Exchange initial matrices */
  for(idx_t m=1; m < nmodes; ++m) {
    mpi_update_rows(ft[m]->indmap, nbr2globs_buf, local2nbr_buf, mats[m],
        globmats[m], rinfo, nfactors, m, DEFAULT_COMM);
  }
#endif

  matrix_t * m1ptr = m1; /* for restoring m1 */

  /* Initialize first A^T * A mats. We redundantly do the first because it
   * makes communication easier. */
  matrix_t * aTa[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    aTa[m] = mat_alloc(nfactors, nfactors);
    mat_aTa(globmats[m], aTa[m], rinfo, thds, nthreads);
  }
  /* used as buffer space */
  aTa[MAX_NMODES] = mat_alloc(nfactors, nfactors);

  /* Compute input tensor norm */
  val_t oldfit = 0;
  val_t fit = 0;
  val_t mynorm = 0;
  #pragma omp parallel for reduction(+:mynorm)
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
  __reset_cpd_timers(rinfo);
  sp_timer_t itertime;
  sp_timer_t modetime[MAX_NMODES];
  timer_start(&timers[TIMER_CPD]);

  idx_t const niters = (idx_t) opts[SPLATT_OPTION_NITER];
  for(idx_t it=0; it < niters; ++it) {
    timer_fstart(&itertime);
    for(idx_t m=0; m < nmodes; ++m) {
      timer_fstart(&modetime[m]);
      mats[MAX_NMODES]->I = ft[0]->dims[m];
      m1->I = globmats[m]->I;
      m1ptr->I = globmats[m]->I;

      /* M1 = X * (C o B) */
      timer_start(&timers[TIMER_MTTKRP]);
      mttkrp_splatt(ft[m], mats, m, thds, nthreads);
      timer_stop(&timers[TIMER_MTTKRP]);
#ifdef SPLATT_USE_MPI
      if(rinfo->distribution > 1 && rinfo->layer_size[m] > 1) {
        m1 = m1ptr;
        /* add my partial multiplications to globmats[m] */
        mpi_add_my_partials(ft[m]->indmap, mats[MAX_NMODES], m1, rinfo,
            nfactors, m);
        /* incorporate neighbors' partials */
        mpi_reduce_rows(local2nbr_buf, nbr2globs_buf, mats[MAX_NMODES], m1,
            rinfo, nfactors, m, DEFAULT_COMM);
      } else {
        /* skip the whole process */
        m1 = mats[MAX_NMODES];
      }
#endif

      /* M2 = (CtC .* BtB .* ...)^-1 */
      __calc_M2(m, nmodes, aTa);

      /* A = M1 * M2 */
      memset(globmats[m]->vals, 0, globmats[m]->I * nfactors * sizeof(val_t));
      mat_matmul(m1, aTa[MAX_NMODES], globmats[m]);

      /* normalize columns and extract lambda */
      if(it == 0) {
        mat_normalize(globmats[m], lambda, MAT_NORM_2, rinfo, thds, nthreads);
      } else {
        mat_normalize(globmats[m], lambda, MAT_NORM_MAX, rinfo, thds,nthreads);
      }

#ifdef SPLATT_USE_MPI
      /* send updated rows to neighbors */
      mpi_update_rows(ft[m]->indmap, nbr2globs_buf, local2nbr_buf, mats[m],
          globmats[m], rinfo, nfactors, m, DEFAULT_COMM);
#endif

      /* update A^T*A */
      mat_aTa(globmats[m], aTa[m], rinfo, thds, nthreads);
      timer_stop(&modetime[m]);
    } /* foreach mode */

    fit = __calc_fit(nmodes, rinfo, thds, nthreads, ttnormsq, lambda,
        globmats, m1, aTa);
    timer_stop(&itertime);

    if(rinfo->rank == 0) {
      printf("  its = %3"SS_IDX" (%0.3fs)  fit = %0.5f  delta = %+0.5f\n",
          it+1, itertime.seconds, fit, fit - oldfit);
      if(opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_LOW) {
        for(idx_t m=0; m < nmodes; ++m) {
          printf("     mode = %1"SS_IDX" (%0.3fs)\n", m+1,
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

  if(rinfo->rank == 0) {
    printf("Final fit: %0.5f\n", fit);
  }

  /* POST PROCESSING */
  /* normalize each mat and adjust lambda */
  val_t * tmp = (val_t *) malloc(nfactors * sizeof(val_t));
  for(idx_t m=0; m < nmodes; ++m) {
    mat_normalize(globmats[m], tmp, MAT_NORM_2, rinfo, thds, nthreads);
    for(idx_t f=0; f < nfactors; ++f) {
      lambda[f] *= tmp[f];
    }
  }
  free(tmp);

  /* CLEAN UP */
  for(idx_t m=0; m < nmodes; ++m) {
    mat_free(aTa[m]);
  }
  mat_free(aTa[MAX_NMODES]);

  thd_free(thds, nthreads);
#ifdef SPLATT_USE_MPI
  if(rinfo->distribution == 3) {
    mat_free(m1ptr);
  }
  free(local2nbr_buf);
  free(nbr2globs_buf);
#endif

#ifdef SPLATT_USE_MPI
  mpi_time_stats(rinfo);
#endif
}



/******************************************************************************
 * CPD OPTIONS FUNCTIONS
 *****************************************************************************/
double * splatt_default_opts(void)
{
  double * opts = (double *) malloc(SPLATT_OPTION_NOPTIONS * sizeof(double));
  for(int i=0; i < SPLATT_OPTION_NOPTIONS; ++i) {
    opts[i] = SPLATT_VAL_OFF;
  }
  opts[SPLATT_OPTION_TOLERANCE] = DEFAULT_TOL;
  opts[SPLATT_OPTION_NITER]     = DEFAULT_ITS;
  opts[SPLATT_OPTION_NTHREADS]  = DEFAULT_THREADS;
  opts[SPLATT_OPTION_TILE]      = SPLATT_NOTILE;
  opts[SPLATT_OPTION_VERBOSITY] = SPLATT_VERBOSITY_LOW;

  return opts;
}

void splatt_free_opts(
  double * opts)
{
  free(opts);
}

