
#include "completion.h"
#include "../csf.h"

#include <math.h>
#include <omp.h>

/* TODO: Conditionally include this OR define lapack prototypes below?
 *       What does this offer beyond prototypes? Can we detect at compile time
 *       if we are using MKL vs ATLAS, etc.?
 */
//#include <mkl.h>



/******************************************************************************
 * LAPACK PROTOTYPES
 *****************************************************************************/

/*
 * TODO: Can this be done in a better way?
 */

#if   SPLATT_VAL_TYPEWIDTH == 32
  void spotrf_(char *, int *, float *, int *, int *);
  void spotrs_(char *, int *, int *, float *, int *, float *, int *, int *);

  #define LAPACK_DPOTRF spotrf_
  #define LAPACK_DPOTRS spotrs_
#else
  void dpotrf_(char *, int *, double *, int *, int *);
  void dpotrs_(char *, int *, int *, double *, int *, double *, int *, int *);
  void dsyrk_(char *, char *, int *, int *, double *, double *, int *, double *, double *, int *);

  #define LAPACK_DPOTRF dpotrf_
  #define LAPACK_DPOTRS dpotrs_
  #define LAPACK_DSYRK dsyrk_
#endif



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Compute the Cholesky decomposition of the normal equations and solve
*        for out_row. We only compute the upper-triangular portion of 'neqs',
*        so work with the lower-triangular portion when column-major
*        (for Fortran).
*
* @param neqs The NxN normal equations.
* @param[out] out_row The RHS of the equation. Updated in place.
* @param N The rank of the problem.
*/
static inline void p_invert_row(
    val_t * const restrict neqs,
    val_t * const restrict out_row,
    idx_t const N)
{
  char uplo = 'L';
  int order = (int) N;
  int lda = (int) N;
  int info;
  LAPACK_DPOTRF(&uplo, &order, neqs, &lda, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRF returned %d\n", info);
  }


  int nrhs = 1;
  int ldb = (int) N;
  LAPACK_DPOTRS(&uplo, &order, &nrhs, neqs, &lda, out_row, &ldb, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRS returned %d\n", info);
  }
}



/**
* @brief Compute DSYRK: out += A^T * A, a rank-k update. Only compute
*        the upper-triangular portion.
*
* @param A The input row(s) to update with.
* @param N The length of 'A'.
* @param nvecs The number of rows in 'A'.
* @param nflush Then number of times this has been performed (this slice).
* @param[out] out The NxN matrix to update.
*/
static inline void p_vec_oprod(
		val_t * const restrict A,
    idx_t const N,
    idx_t const nvecs,
    idx_t const nflush,
    val_t * const restrict out)
{
  char uplo = 'L';
  char trans = 'N';
  int order = (int) N;
  int k = (int) nvecs;
  int lda = (int) N;
  int ldc = (int) N;
  double alpha = 1;
  double beta = (nflush == 0) ? 0. : 1.;
  LAPACK_DSYRK(&uplo, &trans, &order, &k, &alpha, A, &lda, &beta, out, &ldc);
}



static void p_process_tile(
    splatt_csf const * const csf,
    idx_t const tile,
    tc_model * const model,
    tc_ws * const ws,
    thd_info * const thd_densefactors,
    int const tid)
{
  csf_sparsity const * const pt = csf->pt + tile;
  /* empty tile */
  if(pt->vals == 0) {
    return;
  }

  idx_t const nfactors = model->rank;

  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];

  val_t const * const restrict avals = model->factors[csf->dim_perm[1]];
  val_t const * const restrict bvals = model->factors[csf->dim_perm[2]];
  val_t const * const restrict vals = pt->vals;

  /* buffers */
  val_t * const restrict accum = ws->thds[tid].scratch[1];
  val_t * const restrict mat_accum = ws->thds[tid].scratch[3];

  /* update each slice */
  idx_t const nslices = pt->nfibs[0];
  for(idx_t i=0; i < nslices; ++i) {
    /* fid is the row we are actually updating */
    idx_t const fid = (pt->fids[0] == NULL) ? i : pt->fids[0][i];

    /* replicated structures */
    val_t * const restrict out_row =
        (val_t *) thd_densefactors[tid].scratch[0] + (fid * nfactors);
    val_t * const restrict neqs =
        (val_t *) thd_densefactors[tid].scratch[1] + (fid*nfactors*nfactors);

    idx_t bufsize = 0; /* how many hada vecs are in mat_accum */
    idx_t nflush = 1;  /* how many times we have flushed to add to the neqs */
    val_t * restrict hada = mat_accum;

    /* process each fiber */
    for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
      val_t const * const restrict av = avals  + (fids[fib] * nfactors);

      /* first entry of the fiber is used to initialize accum */
      idx_t const jjfirst  = fptr[fib];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accum[r] = vfirst * bv[r];
        hada[r] = av[r] * bv[r];
      }
      hada += nfactors;
      if(++bufsize == ALS_BUFSIZE) {
        /* add to normal equations */
        p_vec_oprod(mat_accum, nfactors, bufsize, nflush++, neqs);
        hada = mat_accum;
        bufsize = 0;
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[fib]+1; jj < fptr[fib+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accum[r] += v * bv[r];
          hada[r] = av[r] * bv[r];
        }
        hada += nfactors;
        if(++bufsize == ALS_BUFSIZE) {
          /* add to normal equations */
          p_vec_oprod(mat_accum, nfactors, bufsize, nflush++, neqs);
          hada = mat_accum;
          bufsize = 0;
        }
      }

      /* accumulate into output row */
      for(idx_t r=0; r < nfactors; ++r) {
        out_row[r] += accum[r] * av[r];
      }
    } /* foreach fiber */

    /* final flush */
    p_vec_oprod(mat_accum, nfactors, bufsize, nflush++, neqs);
  } /* foreach slice */
}



static void p_process_slice3(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const i,
    val_t const * const restrict A,
    val_t const * const restrict B,
    idx_t const nfactors,
    val_t * const restrict out_row,
    val_t * const restrict accum,
    val_t * const restrict neqs,
    val_t * const restrict neqs_buf,
    idx_t * const bufsize,
    idx_t * const nflush)
{
  csf_sparsity const * const pt = csf->pt + tile;
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];
  val_t const * const restrict vals = pt->vals;

  val_t * hada = neqs_buf;

  /* process each fiber */
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    val_t const * const restrict av = A  + (fids[fib] * nfactors);

    /* first entry of the fiber is used to initialize accum */
    idx_t const jjfirst  = fptr[fib];
    val_t const vfirst   = vals[jjfirst];
    val_t const * const restrict bv = B + (inds[jjfirst] * nfactors);
    for(idx_t r=0; r < nfactors; ++r) {
      accum[r] = vfirst * bv[r];
      hada[r] = av[r] * bv[r];
    }

    hada += nfactors;
    if(++(*bufsize) == ALS_BUFSIZE) {
      /* add to normal equations */
      p_vec_oprod(neqs_buf, nfactors, *bufsize, (*nflush)++, neqs);
      *bufsize = 0;
      hada = neqs_buf;
    }

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[fib]+1; jj < fptr[fib+1]; ++jj) {
      val_t const v = vals[jj];
      val_t const * const restrict bv = B + (inds[jj] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accum[r] += v * bv[r];
        hada[r] = av[r] * bv[r];
      }

      hada += nfactors;
      if(++(*bufsize) == ALS_BUFSIZE) {
        /* add to normal equations */
        p_vec_oprod(neqs_buf, nfactors, *bufsize, (*nflush)++, neqs);
        *bufsize = 0;
        hada = neqs_buf;
      }
    }

    /* accumulate into output row */
    for(idx_t r=0; r < nfactors; ++r) {
      out_row[r] += accum[r] * av[r];
    }

  } /* foreach fiber */

  /* final flush */
  p_vec_oprod(neqs_buf, nfactors, *bufsize, (*nflush)++, neqs);
}



/**
* @brief Compute the i-ith row of the MTTKRP, form the normal equations, and
*        store the new row.
*
* @param csf The tensor of training data.
* @param tile The tile that row i resides in.
* @param i The row to update.
* @param reg Regularization parameter for the i-th row.
* @param model The model to update
* @param ws Workspace.
* @param tid OpenMP thread id.
*/
static void p_update_slice(
    splatt_csf const * const csf,
    idx_t const tile,
    idx_t const i,
    val_t const reg,
    tc_model * const model,
    tc_ws * const ws,
    int const tid)
{
  idx_t const nfactors = model->rank;
  csf_sparsity const * const pt = csf->pt + tile;

  assert(model->nmodes == 3);

  /* fid is the row we are actually updating */
  idx_t const fid = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  val_t * const restrict out_row = model->factors[csf->dim_perm[0]] +
      (fid * nfactors);
  val_t * const restrict accum = ws->thds[tid].scratch[1];
  val_t * const restrict neqs  = ws->thds[tid].scratch[2];

  idx_t bufsize = 0; /* how many hada vecs are in mat_accum */
  idx_t nflush = 0;  /* how many times we have flushed to add to the neqs */
  val_t * const restrict mat_accum  = ws->thds[tid].scratch[3];
  val_t * hada = mat_accum;

  for(idx_t f=0; f < nfactors; ++f) {
    out_row[f] = 0;
  }

  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];

  val_t const * const restrict avals = model->factors[csf->dim_perm[1]];
  val_t const * const restrict bvals = model->factors[csf->dim_perm[2]];
  val_t const * const restrict vals = pt->vals;

  /* process each fiber */
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    val_t const * const restrict av = avals  + (fids[fib] * nfactors);

    /* first entry of the fiber is used to initialize accum */
    idx_t const jjfirst  = fptr[fib];
    val_t const vfirst   = vals[jjfirst];
    val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
    for(idx_t r=0; r < nfactors; ++r) {
      accum[r] = vfirst * bv[r];
      hada[r] = av[r] * bv[r];
    }

    hada += nfactors;
    if(++bufsize == ALS_BUFSIZE) {
      /* add to normal equations */
      p_vec_oprod(mat_accum, nfactors, bufsize, nflush++, neqs);
      hada = mat_accum;
      bufsize = 0;
    }

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[fib]+1; jj < fptr[fib+1]; ++jj) {
      val_t const v = vals[jj];
      val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accum[r] += v * bv[r];
        hada[r] = av[r] * bv[r];
      }

      hada += nfactors;
      if(++bufsize == ALS_BUFSIZE) {
        /* add to normal equations */
        p_vec_oprod(mat_accum, nfactors, bufsize, nflush++, neqs);
        bufsize = 0;
        hada = mat_accum;
      }
    }

    /* accumulate into output row */
    for(idx_t r=0; r < nfactors; ++r) {
      out_row[r] += accum[r] * av[r];
    }

  } /* foreach fiber */

  /* final flush */
  p_vec_oprod(mat_accum, nfactors, bufsize, nflush++, neqs);

  /* add regularization to the diagonal */
  for(idx_t f=0; f < nfactors; ++f) {
    neqs[f + (f * nfactors)] += reg;
  }

  /* solve! */
  p_invert_row(neqs, out_row, nfactors);
}



/**
* @brief Update factor[m] which follows a dense mode. This function should be
*        called from inside an OpenMP parallel region!
*
* @param csf The CSF tensor array. csf[m] is a tiled tensor.
* @param m The mode we are updating.
* @param model The current model.
* @param ws Workspace info.
* @param thd_densefactors Thread structures for the dense mode.
* @param tid Thread ID.
*/
static void p_densemode_als_update(
    splatt_csf const * const csf,
    idx_t const m,
    tc_model * const model,
    tc_ws * const ws,
    thd_info * const thd_densefactors,
    int const tid)
{
  idx_t const rank = model->rank;

  /* master thread writes/aggregates directly to the model */
  #pragma omp master
  SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[0], model->factors[m]);

  /* TODO: this could be better by instead only initializing neqs with beta=0
   * and keeping track of which have been updated. */
  memset(thd_densefactors[tid].scratch[0], 0,
      model->dims[m] * rank * sizeof(val_t));
  memset(thd_densefactors[tid].scratch[1], 0,
      model->dims[m] * rank * rank * sizeof(val_t));

  #pragma omp barrier

  /* update each tile in parallel */
  #pragma omp for schedule(dynamic, 1)
  for(idx_t tile=0; tile < csf[m].ntiles; ++tile) {
    p_process_tile(csf+m, tile, model, ws, thd_densefactors, tid);
  }

  /* aggregate partial products */
  thd_reduce(thd_densefactors, 0,
      model->dims[m] * rank, REDUCE_SUM);

  /* TODO: this could be better by using a custom reduction which only
   * operates on the upper triangular portion. OpenMP 4 declare reduction
   * would be good here? */
  thd_reduce(thd_densefactors, 1,
      model->dims[m] * rank * rank, REDUCE_SUM);

  /* save result to model */
  #pragma omp master
  SPLATT_VPTR_SWAP(thd_densefactors[0].scratch[0], model->factors[m]);

  #pragma omp barrier

  /* do all of the Cholesky factorizations */
  val_t * const restrict out  = model->factors[m];
  val_t const reg = ws->regularization[m];
  #pragma omp for schedule(static, 1)
  for(idx_t i=0; i < model->dims[m]; ++i) {
    val_t * const restrict neqs_i =
        (val_t *) thd_densefactors[0].scratch[1] + (i * rank * rank);
    /* add regularization */
    for(idx_t f=0; f < rank; ++f) {
      neqs_i[f + (f * rank)] += reg;
    }

    /* Cholesky + solve */
    p_invert_row(neqs_i, out + (i * rank), rank);
  }
}




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


void splatt_tc_als(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nmodes = train->nmodes;
  idx_t const rank = model->rank;

  /* store dense modes redundantly among threads */
  thd_info * thd_densefactors = NULL;
  if(ws->num_dense > 0) {
    thd_densefactors = thd_init(ws->nthreads, 3,
        ws->maxdense_dim * rank * sizeof(val_t), /* accum */
        ws->maxdense_dim * rank * rank * sizeof(val_t), /* neqs */
        ws->maxdense_dim * sizeof(int)); /* nflush */


    printf("REPLICATING MODES:");
    for(idx_t m=0; m < nmodes; ++m) {
      if(ws->isdense[m]) {
        printf(" %"SPLATT_PF_IDX, m+1);
      }
    }
    printf("\n\n");
  }

  /* load-balanced partition each mode for threads */
  idx_t * parts[MAX_NMODES];

  splatt_csf csf[MAX_NMODES];

  /* convert training data to CSF-ALLMODE */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS] = ws->nthreads;
  for(idx_t m=0; m < nmodes; ++m) {
    if(ws->isdense[m]) {
      /* standard CSF allocation for sparse modes */
      opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE;
      opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
      opts[SPLATT_OPTION_TILEDEPTH] = 1; /* don't tile dense mode */

      csf_alloc_mode(train, CSF_SORTED_MINUSONE, m, csf+m, opts);
      parts[m] = NULL;

    } else {
      /* standard CSF allocation for sparse modes */
      opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE;
      opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
      csf_alloc_mode(train, CSF_SORTED_MINUSONE, m, csf+m, opts);
      parts[m] = csf_partition_1d(csf+m, 0, ws->nthreads);
    }
  }

  val_t prev_val_rmse = 0;

  val_t const loss = tc_loss_sq(train, model, ws);
  val_t const frobsq = tc_frob_sq(model, ws);
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  sp_timer_t mode_timer;
  timer_reset(&mode_timer);
  timer_start(&ws->tc_time);


  for(idx_t e=1; e < ws->max_its+1; ++e) {
    #pragma omp parallel
    {
      int const tid = omp_get_thread_num();

      for(idx_t m=0; m < nmodes; ++m) {
        #pragma omp master
        timer_fstart(&mode_timer);

        if(ws->isdense[m]) {
          p_densemode_als_update(csf, m, model, ws, thd_densefactors, tid);

        /* dense modes are easy */
        } else {
          /* update each row in parallel */
          for(idx_t i=parts[m][tid]; i < parts[m][tid+1]; ++i) {
            p_update_slice(csf+m, 0, i, ws->regularization[m], model, ws, tid);
          }
        }

        #pragma omp barrier

        #pragma omp master
        {
          timer_stop(&mode_timer);
          printf("  mode: %"SPLATT_PF_IDX" time: %0.3fs\n", m+1,
              mode_timer.seconds);
        }
        #pragma omp barrier
      }
    } /* end omp parallel */


    /* compute new obj value, print stats, and exit if converged */
    val_t const loss = tc_loss_sq(train, model, ws);
    val_t const frobsq = tc_frob_sq(model, ws);
    if(tc_converge(train, validate, model, loss, frobsq, e, ws)) {
      break;
    }

  } /* foreach iteration */

  /* cleanup */
  for(idx_t m=0; m < nmodes; ++m) {
    csf_free_mode(csf+m);
    splatt_free(parts[m]);
  }
  if(ws->maxdense_dim > 0) {
    thd_free(thd_densefactors, ws->nthreads);
  }
}



