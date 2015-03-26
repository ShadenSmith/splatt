

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "bench.h"
#include "mttkrp.h"
#include "timer.h"
#include "thd_info.h"
#include "reorder.h"
#include "sort.h"
#include "io.h"
#include "tile.h"

#include <omp.h>

static void __log_mat(
  char const * const ofname,
  matrix_t const * const mat,
  idx_t const * const iperm)
{
  if(iperm != NULL) {
    matrix_t * mat_permed = perm_matrix(mat, iperm, NULL);
    mat_write(mat_permed, ofname);
    mat_free(mat_permed);
  } else {
    mat_write(mat, ofname);
  }
}

static void __shuffle_mats(
  matrix_t ** mats,
  idx_t * const * const perms,
  idx_t const nmodes)
{
  for(idx_t m=0; m < nmodes; ++m) {
    if(perms[m] != NULL) {
      matrix_t * mperm = perm_matrix(mats[m], perms[m], NULL);
      mat_free(mats[m]);
      mats[m] = mperm;
    }
  }
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void bench_splatt(
  sptensor_t * const tt,
  matrix_t ** mats,
  bench_opts const * const opts)
{
  idx_t const niters = opts->niters;
  idx_t const * const threads = opts->threads;
  idx_t const nruns = opts->nruns;
  char matname[64];

  /* shuffle matrices if permutation exists */
  __shuffle_mats(mats, opts->perm->perms, tt->nmodes);

  sp_timer_t itertime;
  sp_timer_t modetime;

  /* add 64 bytes to avoid false sharing */
  thd_info * thds = thd_init(threads[nruns-1], 2,
    mats[0]->J * sizeof(val_t) + 64,
    (mats[0]->J * TILE_SIZES[0] * sizeof(val_t)) + 64);

  ftensor_t * ft[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    ft[m] = ften_alloc(tt, m, opts->tile);
  }
  timer_start(&timers[TIMER_SPLATT]);
  printf("** SPLATT **\n");

  /* for each # threads */
  for(idx_t t=0; t < nruns; ++t) {
    idx_t const nthreads = threads[t];
    omp_set_num_threads(nthreads);
    if(nruns > 1) {
      printf("## THREADS %" SS_IDX "\n", nthreads);
    }

    for(idx_t i=0; i < niters; ++i) {
      timer_fstart(&itertime);
      /* time each mode */
      for(idx_t m=0; m < tt->nmodes; ++m) {
        timer_fstart(&modetime);
        mttkrp_splatt(ft[m], mats, m, thds, nthreads);
        timer_stop(&modetime);
        printf("  mode %" SS_IDX " %0.3fs\n", m+1, modetime.seconds);
        if(opts->write && t == 0 && i == 0) {
          idx_t oldI = mats[MAX_NMODES]->I;
          mats[MAX_NMODES]->I = tt->dims[m];
          sprintf(matname, "splatt_mode%"SS_IDX".mat", m+1);
          __log_mat(matname, mats[MAX_NMODES], opts->perm->iperms[m]);
          mats[MAX_NMODES]->I = oldI;
        }
      }
      timer_stop(&itertime);
      printf("    its = %3"SS_IDX" (%0.3fs)\n", i+1, itertime.seconds);
    }

    /* output load balance info */
    if(nruns > 1 || nthreads > 1) {
      thd_times(thds, threads[nruns-1]);
      thd_reset(thds, threads[nruns-1]);
      printf("\n");
    }
  }
  timer_stop(&timers[TIMER_SPLATT]);

  thd_free(thds, threads[nruns-1]);
  /* clean up */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    ften_free(ft[m]);
  }

  /* fix any matrices that we shuffled */
  __shuffle_mats(mats, opts->perm->iperms, tt->nmodes);
}


void bench_giga(
  sptensor_t * const tt,
  matrix_t ** mats,
  bench_opts const * const opts)
{
  idx_t const niters = opts->niters;
  idx_t const * const threads = opts->threads;
  idx_t const nruns = opts->nruns;
  char matname[64];

  /* shuffle matrices if permutation exists */
  __shuffle_mats(mats, opts->perm->perms, tt->nmodes);

  sp_timer_t itertime;
  sp_timer_t modetime;
  thd_info * thds = thd_init(threads[nruns-1], 0);
  val_t * scratch = (val_t *) malloc(tt->nnz * sizeof(val_t));

  matrix_t * colmats[MAX_NMODES+1];

  printf("** GigaTensor **\n");
  spmatrix_t * unfolds[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    unfolds[m] = tt_unfold(tt, m);
    colmats[m] = mat_mkcol(mats[m]);
  }
  colmats[MAX_NMODES] = mat_mkcol(mats[MAX_NMODES]);

  timer_start(&timers[TIMER_GIGA]);
  for(idx_t t=0; t < nruns; ++t) {
    idx_t const nthreads = threads[t];
    omp_set_num_threads(nthreads);
    if(nruns > 1) {
      printf("## THREADS %"SS_IDX"\n", nthreads);
    }

    for(idx_t i=0; i < niters; ++i) {
      timer_fstart(&itertime);
      for(idx_t m=0; m < tt->nmodes; ++m) {
        timer_fstart(&modetime);
        mttkrp_giga(unfolds[m], colmats, m, scratch);
        timer_stop(&modetime);
        printf("  mode %"SS_IDX" %0.3fs\n", m+1, modetime.seconds);
        if(opts->write && t == 0 && i == 0) {
          colmats[MAX_NMODES]->I = tt->dims[m];
          sprintf(matname, "giga_mode%"SS_IDX".mat", m+1);
          __log_mat(matname, colmats[MAX_NMODES], opts->perm->iperms[m]);
        }
      }
      timer_stop(&itertime);
      printf("    its = %3"SS_IDX" (%0.3fs)\n", i+1, itertime.seconds);
    }

    /* output load balance info */
    if(nruns > 1 || nthreads > 1) {
      thd_times(thds, threads[nruns-1]);
      thd_reset(thds, threads[nruns-1]);
      printf("\n");
    }
  }
  timer_stop(&timers[TIMER_GIGA]);

  thd_free(thds, threads[nruns-1]);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    spmat_free(unfolds[m]);
    mat_free(colmats[m]);
  }
  mat_free(colmats[MAX_NMODES]);
  free(scratch);

  /* fix any matrices that we shuffled */
  __shuffle_mats(mats, opts->perm->iperms, tt->nmodes);
}


void bench_ttbox(
  sptensor_t * const tt,
  matrix_t ** mats,
  bench_opts const * const opts)
{
  idx_t const niters = opts->niters;
  idx_t const * const threads = opts->threads;
  idx_t const nruns = opts->nruns;
  char matname[64];

  tt_sort(tt, 0, NULL);

  /* shuffle matrices if permutation exists */
  __shuffle_mats(mats, opts->perm->perms, tt->nmodes);

  sp_timer_t itertime;
  sp_timer_t modetime;

  matrix_t * colmats[MAX_NMODES+1];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    colmats[m] = mat_mkcol(mats[m]);
  }
  colmats[MAX_NMODES] = mat_mkcol(mats[MAX_NMODES]);

  thd_info * thds = thd_init(threads[nruns-1], 0);

  printf("** TTBOX **\n");
  val_t * scratch = (val_t *) malloc(tt->nnz * sizeof(val_t));

  timer_start(&timers[TIMER_TTBOX]);
  for(idx_t t=0; t < nruns; ++t) {
    idx_t const nthreads = threads[t];
    omp_set_num_threads(nthreads);
    if(nruns > 1) {
      printf("## THREADS %"SS_IDX"\n", nthreads);
    }

    for(idx_t i=0; i < niters; ++i) {
      timer_fstart(&itertime);
      for(idx_t m=0; m < tt->nmodes; ++m) {
        timer_fstart(&modetime);
        mttkrp_ttbox(tt, colmats, m, scratch);
        timer_stop(&modetime);
        printf("  mode %"SS_IDX" %0.3fs\n", m+1, modetime.seconds);
        if(opts->write && t == 0 && i == 0) {
          colmats[MAX_NMODES]->I = tt->dims[m];
          sprintf(matname, "ttbox_mode%"SS_IDX".mat", m+1);
          __log_mat(matname, colmats[MAX_NMODES], opts->perm->iperms[m]);
        }
      }
      timer_stop(&itertime);
      printf("    its = %3"SS_IDX" (%0.3fs)\n", i+1, itertime.seconds);
    }


    /* output load balance info */
    if(nruns > 1 || nthreads > 1) {
      thd_times(thds, threads[nruns-1]);
      thd_reset(thds, threads[nruns-1]);
      printf("\n");
    }
  }
  timer_stop(&timers[TIMER_TTBOX]);

  thd_free(thds, threads[nruns-1]);
  free(scratch);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mat_free(colmats[m]);
  }
  mat_free(colmats[MAX_NMODES]);

  /* fix any matrices that we shuffled */
  __shuffle_mats(mats, opts->perm->iperms, tt->nmodes);
}


