/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "matrix.h"
#include "sptensor.h"
#include "stats.h"
#include "timer.h"
#include "thd_info.h"
#include "tile.h"
#include "io.h"
#include "util.h"
#include "ttm.h"

#include "tucker.h"
#include "svd.h"

#include <omp.h>
#include <math.h>



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
int splatt_tucker_als(
    splatt_idx_t const * const nfactors,
    splatt_idx_t const nmodes,
    splatt_csf const * const tensors,
    double const * const options,
    splatt_tucker_t * factored)
{
  matrix_t * mats[MAX_NMODES+1];

  /* fill in factored */
  idx_t maxcols = 0;
  idx_t csize = 1;
  factored->nmodes = nmodes;
  for(idx_t m=0; m < nmodes; ++m) {
    factored->rank[m] = nfactors[m];
    mats[m] = mat_rand(tensors[0].dims[m], nfactors[m]);
    factored->factors[m] = mats[m]->vals;

    csize *= nfactors[m];
    maxcols = SS_MAX(maxcols, nfactors[m]);
  }
  factored->core = (val_t *) calloc(csize, sizeof(val_t));

  /* print Tucker stats? */
  splatt_verbosity_type which_verb = options[SPLATT_OPTION_VERBOSITY];
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    tucker_stats(tensors, nfactors, options);
  }

  /* compute the factorization */
  tucker_hooi_iterate(tensors, mats, factored->core, nfactors, options);

  /* cleanup */
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]); /* just delete the pointer */
  }

  return SPLATT_SUCCESS;
}


void splatt_free_tucker(
    splatt_tucker_t * factored)
{
  free(factored->core);
  for(idx_t m=0; m < factored->nmodes; ++m) {
    free(factored->factors[m]);
  }
}




/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

typedef struct
{
  /* state tracking */
  idx_t idxstack[MAX_NMODES];

  /** The number of columns in the output of TTMc (for each mode) */
  idx_t gten_cols[MAX_NMODES];

  /** The size of the outer product at each depth during TTMc (for each mode).
   *  This routine accounts for CSF-ALLOC strategy, because the size of outer
   *  products will vary based on the CSF traversal strategy.
   *
   *  Examples:
   *  1. accum_size[0][:] gives the size of each accumulation during TTMc for
   *  the first mode.
   *
   *  2. accum_size[1][nmodes-1] gives the size of the accumulation just above
   *  the leaf level during the second TTMc operation.
   */
  idx_t accum_size[MAX_NMODES][MAX_NMODES];

  /* buffers */
  idx_t * accum_fids[MAX_NMODES];
  val_t * accum[MAX_NMODES];

  /* SVD */
  svd_ws sws;
} tucker_ws;



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Compute the number of columns in the TTMc output for all modes. The
*        total core tensor size also is written to ncols[nmodes].
*
* @param nfactors The rank of the decomposition in each mode.
* @param nmodes The number of modes.
* @param[out] ncols ncols[m] stores the number of columns in the output of the
*                   mode-m TTMc.
*/
static void p_compute_ncols(
    idx_t const * const nfactors,
    idx_t const nmodes,
    idx_t * const ncols)
{
  /* initialize */
  for(idx_t m=0; m <= nmodes; ++m) {
    ncols[m] = 1;
  }

  /* fill in all modes, plus ncols[nmodes] which stores core size */
  for(idx_t m=0; m <= nmodes; ++m) {
    for(idx_t moff=0; moff < nmodes; ++moff) {
      /* skip the mode we are computing */
      if(moff != m) {
        ncols[m] *= nfactors[moff];
      }
    }
  }
}


/**
* @brief Fill an array with the mode permutation used to compute a Tucker core.
*
* @param tensors The CSF tensor.
* @param[out] perm The permutation array to fill.
* @param opts The options used to allocate 'tensors'
*/
static void p_fill_core_perm(
    splatt_csf const * const tensors,
    idx_t * const perm,
    double const * const opts)
{
  idx_t const nmodes = tensors->nmodes;

  splatt_csf_type const which = opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {
  case SPLATT_CSF_ONEMODE:
    memcpy(perm, tensors[0].dim_perm, nmodes * sizeof(*perm));
    break;

  case SPLATT_CSF_TWOMODE:
    memcpy(perm, tensors[1].dim_perm, nmodes * sizeof(*perm));
    break;

  case SPLATT_CSF_ALLMODE:
    memcpy(perm, tensors[nmodes-1].dim_perm, nmodes * sizeof(*perm));
    break;

  default:
    /* XXX */
    fprintf(stderr, "SPLATT: splatt_csf_type %d not recognized.\n", which);
    break;
  }
}


static void p_print_cache_size(
    tucker_ws const * const ws,
    splatt_csf const * const tensors,
    idx_t const * const nfactors,
    double const * const opts)
{
  idx_t const nmodes = tensors->nmodes;
  for(idx_t m=0; m < nmodes; ++m) {
    splatt_csf const * const csf = tensors;

    idx_t mult = ws->gten_cols[m] / nfactors[m];
    for(idx_t depth = 1; depth < nmodes - 1; ++depth) {
      idx_t bytes = 0;
      idx_t nfibs = 0;

      for(idx_t t=0; t < csf->ntiles; ++t) {
        csf_sparsity const * const pt = csf->pt + t;
        bytes += pt->nfibs[depth] * nfactors[csf->dim_perm[depth]] * mult;
        nfibs += pt->nfibs[depth];
      }

      /* depth computed */
      char * s = bytes_str(bytes * sizeof(val_t));
      printf("depth %"SPLATT_PF_IDX" nfibs=%"SPLATT_PF_IDX" (%s)\n", depth,
          nfibs, s);
      free(s);

      mult /= nfactors[csf->dim_perm[depth]];
    }
  }
}



static void p_print_cache_size2(
    tucker_ws const * const ws,
    splatt_csf const * const csf,
    idx_t const * const nfactors,
    double const * const opts)
{
  idx_t const nmodes = csf->nmodes;
  idx_t ncols = ws->gten_cols[0] / nfactors[0];
  for(idx_t depth = 1; depth < nmodes - 1; ++depth) {
    idx_t bytes = 0;
    idx_t nfibs = 0;

    for(idx_t t=0; t < csf->ntiles; ++t) {
      csf_sparsity const * const pt = csf->pt + t;
      printf("ncols: %lu\n", ncols);
      bytes += pt->nfibs[depth] * ncols;
      nfibs += pt->nfibs[depth];
    }

    /* depth computed */
    char * s = bytes_str(bytes * sizeof(val_t));
    printf("depth %"SPLATT_PF_IDX" nfibs=%"SPLATT_PF_IDX" (%0.2f%% nnz) (%s)\n",
        depth, nfibs, 100. * ((double) nfibs / (double) csf->nnz), s);
    free(s);

    ncols /= nfactors[csf->dim_perm[depth]];
  }
}




/**
* @brief Allocate and fill a Tucker workspace.
*
* @param[out] ws The workspace to fill.
* @param nfactors The ranks of the decomposition.
* @param tensors The CSF tensor(s).
* @param opts The options used during CSF allocation and factorization.
*/
static void p_alloc_tucker_ws(
    tucker_ws * const ws,
    idx_t const * const nfactors,
    splatt_csf const * const tensors,
    double const * const opts)
{
  idx_t const nmodes = tensors->nmodes;

  /* initialize ptr arrays with NULL */
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    ws->accum_fids[m] = NULL;
    ws->accum[m] = NULL;
  }

  /* fill #cols in TTMc output */
  p_compute_ncols(nfactors, nmodes, ws->gten_cols);

  p_print_cache_size2(ws, tensors, nfactors, opts);
  return;

  /* SVD allocations */
  alloc_svd_ws(&(ws->sws), nmodes, tensors->dims, ws->gten_cols);
}



/**
* @brief Free all memory allocated for a Tucker workspace.
*
* @param ws The workspace to free.
*/
static void p_free_tucker_ws(
    tucker_ws * const ws)
{
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    free(ws->accum_fids[m]);
    free(ws->accum[m]);
  }
  free_svd_ws(&(ws->sws));
}






/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

double tucker_calc_fit(
    val_t const * const core,
    idx_t const core_size,
    val_t const ttnormsq)
{
  timer_start(&timers[TIMER_FIT]);

  val_t gnormsq = 0;
  for(idx_t x=0; x < core_size; ++x) {
    gnormsq += core[x] * core[x];
  }

  double const residual = sqrt(ttnormsq - gnormsq);
  double fit = 1 - (residual / sqrt((double)ttnormsq));

  timer_stop(&timers[TIMER_FIT]);

  return fit;
}


thd_info * tucker_alloc_thds(
    idx_t const nthreads,
    splatt_csf const * const tensors,
    idx_t const * const nfactors,
    double const * const opts)
{
  idx_t const nmodes = tensors->nmodes;

  /* find largest number of fibers we need to accumulate */
  idx_t largest_nfibs[MAX_NMODES];
  ttmc_largest_outer(tensors, largest_nfibs, opts);
  idx_t const largest = largest_nfibs[argmax_elem(largest_nfibs, nmodes)];

  /* find # columns for each TTMc and output core */
  idx_t ncols[MAX_NMODES+1];
  p_compute_ncols(nfactors, nmodes, ncols);
  idx_t const maxcols = ncols[argmax_elem(ncols, nmodes)];

  thd_info * thds =  thd_init(nthreads, 3,
    /* nnz accumulation */
    (largest * maxcols * sizeof(val_t)) + 64,
    /* fids */
    (largest * sizeof(idx_t)) + 64,
    /* actual rows corresponding to fids */
    (tenout_dim(nmodes, nfactors, largest_nfibs) * sizeof(val_t)) + 64);

  return thds;
}


void make_core(
    val_t * ttmc,
    val_t * lastmat,
    val_t * core,
    idx_t const nmodes,
    idx_t const mode,
    idx_t const * const nfactors,
    idx_t const nlongrows)
{
	timer_start(&timers[TIMER_MATMUL]);
  idx_t ncols = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    if(m != mode) {
      ncols *= nfactors[m];
    }
  }

  /* C = A' * B */
  val_t * A = lastmat;
  val_t * B = ttmc;
  val_t * C = core;

  int M = nfactors[mode];
  int N = ncols;
  int K = nlongrows;

  char transA = 'T';
  char transB = 'N';
  val_t alpha = 1.;
  val_t beta = 0;

  /* C' = B' * A, but transA/B are flipped to account for row-major ordering */
	BLAS_GEMM(
      &transB, &transA,
      &N, &M, &K,
      &alpha,
      B, &N,
      A, &M,
      &beta,
      C, &N);

	timer_stop(&timers[TIMER_MATMUL]);
}


double tucker_hooi_iterate(
    splatt_csf const * const tensors,
    matrix_t ** mats,
    val_t * const core,
    idx_t const * const nfactors,
    double const * const opts)
{
  idx_t const nmodes = tensors->nmodes;

  tucker_ws ws;
  p_alloc_tucker_ws(&ws, nfactors, tensors, opts);
  return SPLATT_SUCCESS;

  /* allocate the TTMc output */
  idx_t const tenout_size = tenout_dim(nmodes, nfactors, tensors->dims);
  val_t * gten = malloc(tenout_size * sizeof(*gten));

  /* find # columns for each TTMc and output core */
  idx_t ncols[MAX_NMODES+1];
  p_compute_ncols(nfactors, nmodes, ncols);

  /* thread structures */
  idx_t const nthreads = (idx_t) opts[SPLATT_OPTION_NTHREADS];
  omp_set_num_threads(nthreads);
  thd_info * thds =  tucker_alloc_thds(nthreads, tensors, nfactors, opts);

  sp_timer_t itertime;
  sp_timer_t modetime[MAX_NMODES];

  double oldfit = 0;
  double fit = 0;

  val_t const ttnormsq = csf_frobsq(tensors);

  /* foreach iteration */
  idx_t const niters = (idx_t) opts[SPLATT_OPTION_NITER];
  for(idx_t it=0; it < niters; ++it) {
    timer_fstart(&itertime);

    /* foreach mode */
    for(idx_t m=0; m < nmodes; ++m) {
      timer_fstart(&modetime[m]);

      timer_start(&timers[TIMER_TTM]);
      ttmc_csf(tensors, mats, gten, m, thds, opts);
      timer_stop(&timers[TIMER_TTM]);

      /* find the truncated SVD of the TTMc output */
      left_singulars(gten, mats[m]->vals, mats[m]->I, ncols[m], mats[m]->J,
          &(ws.sws));

      timer_stop(&modetime[m]);
    }
    timer_stop(&itertime);

    /* compute core */
    make_core(gten, mats[nmodes-1]->vals, core, nmodes, nmodes-1, nfactors,
        tensors->dims[nmodes-1]);

    /* check for convergence */
    fit = tucker_calc_fit(core, ncols[nmodes], ttnormsq);

    /* print progress */
    if(opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_NONE) {
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

  } /* foreach iteration */

  p_free_tucker_ws(&ws);
  free(gten);
  thd_free(thds, nthreads);

  permute_core(tensors, core, nfactors, opts);

  return fit;
}


void permute_core(
    splatt_csf const * const tensors,
    val_t * const core,
    idx_t const * const nfactors,
    double const * const opts)
{
  idx_t const nmodes = tensors->nmodes;
  idx_t ncols[MAX_NMODES+1];
  p_compute_ncols(nfactors, nmodes, ncols);

  idx_t perm[MAX_NMODES];
  p_fill_core_perm(tensors, perm, opts);

  val_t * newcore = malloc(ncols[nmodes] * sizeof(*newcore));

  idx_t ind[MAX_NMODES];
  for(idx_t x=0; x < ncols[nmodes]; ++x) {
    /* translate x into ind */
    idx_t id = x;
    for(idx_t m=nmodes; m-- != 0; ){
      ind[m] = id % nfactors[m];
      id /= nfactors[m];
    }

    /* translate ind into an index into core */
    idx_t mult = ncols[nmodes-1];
    idx_t translated = mult * ind[perm[0]];
    for(idx_t m=1; m < nmodes; ++m) {
      mult /= nfactors[perm[m]];
      translated += mult * ind[perm[m]];
    }

    /* now copy */
    newcore[x] = core[translated];
  }

  /* copy permuted core into old core */
  memcpy(core, newcore, ncols[nmodes] * sizeof(*core));
  free(newcore);
}


