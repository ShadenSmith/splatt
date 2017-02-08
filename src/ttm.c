
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "ttm.h"
#include "thd_info.h"
#include "tile.h"
#include "util.h"


#include "mutex_pool.h"


/* XXX: this is a memory leak */
static mutex_pool * pool = NULL;



/**
* @brief The number of rows we will buffer for DGEMM when propagating up
*        partial computations of TTMc (sum of outer products).
*/
static idx_t const TTMC_BUFROWS = 128;



/* Count FLOPS during ttmc */
#ifndef SPLATT_TTMC_FLOPS
#define SPLATT_TTMC_FLOPS 0

static size_t nflops = 0;
#endif

#ifndef SPLATT_TTMC_PREFETCH
#define SPLATT_TTMC_PREFETCH 0
#endif


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

int splatt_ttmc(
    splatt_idx_t const mode,
    splatt_idx_t const * const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const tenout,
    double const * const options)
{
  idx_t const nmodes = tensors->nmodes;

  idx_t maxcols = 0;

  /* fill matrix pointers  */
  matrix_t * mats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (matrix_t *) malloc(sizeof(matrix_t));
    mats[m]->I = tensors->dims[m];
    mats[m]->J = ncolumns[m],
    mats[m]->rowmajor = 1;
    mats[m]->vals = matrices[m];

    maxcols = SS_MAX(maxcols, ncolumns[m]);
  }

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  idx_t const nthreads = (idx_t) options[SPLATT_OPTION_NTHREADS];
  splatt_omp_set_num_threads(nthreads);
  thd_info * thds = ttmc_alloc_thds(nthreads, tensors, ncolumns, options);

  ttmc_csf(tensors, mats, tenout, mode, thds, options);

  /* cleanup */
  thd_free(thds, nthreads);
  for(idx_t m=0; m < nmodes; ++m) {
    splatt_free(mats[m]);
  }

  return SPLATT_SUCCESS;
}



int splatt_ttmc_full(
    splatt_idx_t const * const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const tenout,
    double const * const options)
{
  idx_t const nmodes = tensors->nmodes;

  idx_t const nrows = tensors->dims[nmodes-1];
  idx_t ncols = 1;
  for(idx_t m=0; m < nmodes-1; ++m) {
    ncols *= ncolumns[m];
  }

  /* TTMc with all but the last mode. */
  val_t * ttmc_buf = splatt_malloc(nrows * ncols * sizeof(*ttmc_buf));

  splatt_ttmc(nmodes-1, ncolumns, tensors, matrices, ttmc_buf, options);

  /* Multiply with the last mode. */
  make_core(ttmc_buf, matrices[nmodes-1], tenout, nmodes, nmodes-1, ncolumns,
      nrows);
  splatt_free(ttmc_buf);

  /* Permute to invert tensors[nmodes-1].dim_perm. */
  permute_core(tensors, tenout, ncolumns, options);

  return SPLATT_SUCCESS;
}


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


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
    /* if longest mode was the last mode */
    if(nmodes-1 == tensors[0].dim_perm[nmodes-1]) {
      memcpy(perm, tensors[1].dim_perm, nmodes * sizeof(*perm));
    } else {
      memcpy(perm, tensors[0].dim_perm, nmodes * sizeof(*perm));
    }
    break;

  case SPLATT_CSF_ALLMODE:
    memcpy(perm, tensors[nmodes-1].dim_perm, nmodes * sizeof(*perm));
    break;

  default:
    /* XXX */
    fprintf(stderr, "SPLATT: splatt_csf_type %d not recognized.\n", which);
    break;
  }

  /* If we did not compute the last mode on root, adjust permutation by moving
   * the last computed mode to front. */
  if(perm[0] != nmodes-1) {
    for(idx_t m=0; m < nmodes; ++m) {
      /* move last mode to beginning and shift to fit */
      if(perm[m] == nmodes-1) {
        memmove(perm+1, perm, m * sizeof(*perm));
        perm[0] = nmodes-1;
      }
    }
  }
}

/**
* @brief Count the number of columns in the output of ttmc with 'mode'
*
* @param nfactors The # columns in each of the factors.
* @param nmodes The number of modes in the factorization (and length of
*               'nfactors').
* @param mode The mode we are interested in.
*
* @return The prod(nfactors[:]) / nfactors[mode].
*/
static inline idx_t p_ttmc_outncols(
    idx_t const * const nfactors,
    idx_t const nmodes,
    idx_t const mode)
{
  /* total size of output */
  idx_t ncols = 1;
  for(idx_t m = 0; m < nmodes; ++m) {
    if(m != mode) {
      ncols *= nfactors[m];
    }
  }
  return ncols;
}


/**
* @brief Compute (rowA^T rowB) into row-major 'out'. Any former values in 'out'
*        are overwritten.
*
* @param rowA The first row vector.
* @param nA The number of elements in rowA.
* @param rowB The second row vector.
* @param nB The number of elements in rowB.
* @param[out] out The output matrix which is (nA x nB).
*/
static inline void p_twovec_outer_prod(
    val_t const * const restrict rowA,
    idx_t const nA,
    val_t const * const restrict rowB,
    idx_t const nB,
    val_t * const restrict out)
{
  for(idx_t i=0; i < nA; ++i) {
    val_t * const restrict orow = out + (i*nB);
    val_t const ival = rowA[i];
    for(idx_t j=0; j < nB; ++j) {
      orow[j] = ival * rowB[j];
    }
  }
}

/**
* @brief Compute (rowA^T rowB) AND ACCUMULATE into row-major 'out'.
*
* @param rowA The first row vector.
* @param nA The number of elements in rowA.
* @param rowB The second row vector.
* @param nB The number of elements in rowB.
* @param[out] out The output matrix which is (nA x nB).
*/
static inline void p_twovec_outer_prod_accum(
    val_t const * const restrict rowA,
    idx_t const nA,
    val_t const * const restrict rowB,
    idx_t const nB,
    val_t * const restrict out)
{
  for(idx_t i=0; i < nA; ++i) {
    val_t * const restrict orow = out + (i*nB);
    val_t const ival = rowA[i];
    for(idx_t j=0; j < nB; ++j) {
      orow[j] += ival * rowB[j];
    }
  }
}


/**
* @brief Compute C = A * B^T, where A is a column-major matrix of the gathered
*        rows related to fiber ids. B is a column-major matrix of the
*        accumulated nonzero values.
*
* @param fids_buf The gathered rows of the factor matrix (from fiber ids).
* @param ncol_fids The number of columns in the gathered rows, or the number of
*                  rows in the column-major matrix.
* @param accums_buf The accumulated nonzero contributions.
* @param ncol_accums The number of columns in the accumulation.
* @param nfibers The number of total fibers (outer products) to work on.
* @param[out] out The output matrix, which is (ncol_fids x ncol_accums).
*/
static inline void p_twovec_outer_prod_tiled(
    val_t const * const restrict fids_buf,
    idx_t const ncol_fids,
    val_t const * const restrict accums_buf,
    idx_t const ncol_accums,
    idx_t const nfibers,
    val_t * const restrict out)
{
#if 1
  val_t alpha = 1.;
  val_t beta = 1.;
  char transA = 'N';
  char transB = 'T';
  int M = ncol_accums;
  int N = ncol_fids;
  int K = nfibers;

	BLAS_GEMM(
      &transA, &transB, /* transposes */
      &M, &N, &K,       /* dimensions */
      &alpha,
      accums_buf, &M,   /* A */
      fids_buf, &N,     /* B */
      &beta, out, &M);  /* C */
#else
  for(idx_t k=0; k < nfibers; ++k) {
    p_twovec_outer_prod_accum(
        fids_buf+(k*ncol_fids), ncol_fids,
        accums_buf+(k*ncol_accums), ncol_accums,
        out);
  }
#endif
}




/**
* @brief Flush buffered rows during TTMc. This requires gathering factor rows
*        into a contiguous buffer, followed by a DGEMM call.
*
* @param row_buffer The buffer we will gather into. Must be at least
*                   (naccum x factor_ncols) in size.
* @param row_ids The row numbers we will gather.
* @param factor The factor matrix to access (whose rows we gather).
* @param factor_ncols The number of columns in 'factor'.
* @param accums_buf The accumulated partial products.
* @param ncol_accums The number of columns in 'accums_buf' partials.
* @param naccum The number of rows in 'accums_buf' and 'row_ids'.
* @param[out] out We accumulate into this row.
*/
static inline void p_flush_oprods(
    val_t       * const row_buffer,
    idx_t const * const restrict row_ids,
    val_t const * const factor,
    idx_t const factor_ncols,
    val_t const * const restrict accums_buf,
    idx_t const ncol_accums,
    idx_t const naccum,
    val_t * const restrict out)
{
  /* no-op */
  if(naccum == 0) {
    return;
  }

#if SPLATT_TTMC_PREFETCH == 1
    __builtin_prefetch(factor + (row_ids[0] * factor_ncols), 0, 0);
#endif

#if SPLATT_TTMC_FLOPS == 1
  #pragma omp atomic
  nflops += 2 * (factor_ncols + ncol_accums * naccum);
#endif

  /* don't do the gather step or call to BLAS_GEMM */
  if(naccum == 1) {
    p_twovec_outer_prod_accum(factor + (row_ids[0] * factor_ncols),
                               factor_ncols, accums_buf, ncol_accums, out);
    return;
  }

  /* gather rows into row_buffer */
  for(idx_t row=0; row < naccum-1; ++row) {
#if SPLATT_TTMC_PREFETCH == 1
    __builtin_prefetch(factor + (row_ids[row+1] * factor_ncols), 0, 0);
#endif

    val_t       * const restrict accum = row_buffer + (row * factor_ncols);
    val_t const * const restrict av = factor + (row_ids[row] * factor_ncols);
    for(idx_t f=0; f < factor_ncols; ++f) {
      accum[f] = av[f];
    }
  }

  /* last accum */
  val_t       * const restrict accum = row_buffer + ((naccum-1) * factor_ncols);
  val_t const * const restrict av = factor + (row_ids[naccum-1] * factor_ncols);
  for(idx_t f=0; f < factor_ncols; ++f) {
    accum[f] = av[f];
  }

  /* dgemm */
  p_twovec_outer_prod_tiled(row_buffer, factor_ncols, accums_buf, ncol_accums,
      naccum, out);
}





/**
* @brief Calculate the size of the output tensor and zero it.
*
* @param tenout The output tensor.
* @param mats The input matrices (they determine the size of the output).
* @param nmodes The number of modes.
* @param mode The mode we are computing.
* @param dims The dimensions of the input tensor.
*/
static inline void p_clear_tenout(
    val_t * const restrict tenout,
    matrix_t ** mats,
    idx_t const nmodes,
    idx_t const mode,
    idx_t const * const dims)
{
  /* clear tenout */
  idx_t outsize = dims[mode];
  for(idx_t m=0; m < nmodes; ++m) {
    if(m != mode) {
      outsize *= mats[m]->J;
    }
  }
  par_memset(tenout, 0, outsize * sizeof(*tenout));
}


static inline void p_csf_process_fiber(
  val_t * const restrict accumbuf,
  idx_t const nfactors,
  val_t const * const leafmat,
  idx_t const start,
  idx_t const end,
  idx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  /* foreach nnz in fiber */
  for(idx_t j=start; j < end; ++j) {
    val_t const v = vals[j] ;
    val_t const * const restrict row = leafmat + (nfactors * inds[j]);
    for(idx_t f=0; f < nfactors; ++f) {
      accumbuf[f] += v * row[f];
    }
  }
#if SPLATT_TTMC_FLOPS == 1
    #pragma omp atomic
    nflops += 2 * nfactors * (end - start);
#endif
}




static inline void p_csf_process_fiber_lock(
  val_t * const tenout,
  val_t const * const restrict accumbuf,
  idx_t const ncols,
  idx_t const start,
  idx_t const end,
  idx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = tenout + (inds[jj] * ncols);
    val_t const v = vals[jj];

    mutex_set_lock(pool, inds[jj]);
    for(idx_t f=0; f < ncols; ++f) {
      leafrow[f] += v * accumbuf[f];
    }
    mutex_unset_lock(pool, inds[jj]);
  }
}




static inline void p_propagate_up(
  val_t * const out,
  val_t * const * const buf,
  idx_t * const restrict idxstack,
  idx_t const init_depth,
  idx_t const init_idx,
  idx_t const * const * const fp,
  idx_t const * const * const fids,
  val_t const * const restrict vals,
  val_t ** mvals,
  idx_t const nmodes,
  idx_t const * const restrict ncols_mat,
  idx_t const * const restrict ncols_lvl)
{
  /* push initial idx initialize idxstack */
  assert(init_depth < nmodes-1);
  idxstack[init_depth] = init_idx;
  for(idx_t m=init_depth+1; m < nmodes; ++m) {
    idxstack[m] = fp[m-1][idxstack[m-1]];
  }

  /* clear out accumulation buffer */
  for(idx_t f=0; f < ncols_lvl[init_depth+1]; ++f) {
    buf[init_depth+1][f] = 0;
  }

  while(idxstack[init_depth+1] < fp[init_depth][init_idx+1]) {
    /* skip to last internal mode */
    idx_t depth = nmodes - 2;

    assert(ncols_lvl[depth+1] == ncols_mat[depth+1]);
    /* process all nonzeros [start, end) into buf[depth]*/
    idx_t const start = fp[depth][idxstack[depth]];
    idx_t const end   = fp[depth][idxstack[depth]+1];
    p_csf_process_fiber(buf[depth+1], ncols_mat[depth+1], mvals[depth+1],
        start, end, fids[depth+1], vals);

    idxstack[depth+1] = end;

    /* exit early if there is no propagation to do... */
    if(init_depth == nmodes-2) {
      break;
    }

    /* Propagate up until we reach a node with more children to process */
    do {
      /* propagate result up and clear buffer for next sibling */
      val_t const * const restrict fibrow
          = mvals[depth] + (fids[depth][idxstack[depth]] * ncols_mat[depth]);
      assert(ncols_lvl[depth] == ncols_lvl[depth+1] * ncols_mat[depth]);
      p_twovec_outer_prod_accum(
          fibrow, ncols_mat[depth],
          buf[depth+1], ncols_lvl[depth+1],
          buf[depth]);
#if SPLATT_TTMC_FLOPS == 1
      #pragma omp atomic
      nflops += 2 * (ncols_mat[depth] * ncols_lvl[depth+1]);
#endif
      /* clear */
      for(idx_t f=0; f < ncols_lvl[depth+1]; ++f) {
        buf[depth+1][f] = 0.;
      }

      ++idxstack[depth];
      --depth;
    } while(depth > init_depth &&
        idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
  } /* end DFS */

  /* copy to out */
  for(idx_t f=0; f < ncols_lvl[init_depth+1]; ++f) {
    out[f] = buf[init_depth+1][f];
  }
}




/******************************************************************************
 * TRAVERSAL FUNCTIONS
 *****************************************************************************/



/**
* @brief Perform TTM on the root mode of a 3D CSF tensor. No locks are used.
*
* @param csf The input tensor.
* @param tile_id Which tile.
* @param mats Input matrices.
* @param tenout Output tensor.
* @param thds Thread structures.
*/
static void p_csf_ttmc_root3(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{
  assert(csf->nmodes == 3);

  matrix_t const * const A = mats[csf->dim_perm[1]];
  matrix_t const * const B = mats[csf->dim_perm[2]];

  idx_t const rankA = A->J;
  idx_t const rankB = B->J;

  val_t const * const restrict vals = csf->pt[tile_id].vals;

  idx_t const * const restrict sptr = csf->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = csf->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = csf->pt[tile_id].fids[0];
  idx_t const * const restrict fids = csf->pt[tile_id].fids[1];
  idx_t const * const restrict inds = csf->pt[tile_id].fids[2];

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  int const tid = omp_get_thread_num();
  val_t * const restrict accum_nnz_raw = (val_t *) thds[tid].scratch[0];

  /* buffered outer products */
  idx_t naccum;
  idx_t * const accum_fids  = thds[tid].scratch[1];
  val_t * const accum_oprod = thds[tid].scratch[2];

  /* foreach slice */
  idx_t const nslices = csf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    val_t * const restrict outv = tenout + (fid * rankA * rankB);

    naccum = 0;

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* grab nnz accumulation buffer */
      val_t * const restrict accum_nnz = accum_nnz_raw + (naccum * rankB);

      /* first entry of the fiber is used to initialize accum_nnz */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * rankB);
      for(idx_t r=0; r < rankB; ++r) {
        accum_nnz[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * rankB);
        for(idx_t r=0; r < rankB; ++r) {
          accum_nnz[r] += v * bv[r];
        }
      }
#if SPLATT_TTMC_FLOPS == 1
      #pragma omp atomic
      nflops += 2 * rankB * (fptr[f+1] - fptr[f]);
#endif


      /* accumulate outer product into tenout */
      accum_fids[naccum++] = fids[f];
      if((TTMC_BUFROWS == 1) || (naccum == TTMC_BUFROWS)) {
        p_flush_oprods(accum_oprod, accum_fids, avals, rankA,
                       accum_nnz_raw, rankB, naccum, outv);
        naccum = 0;
      }
    } /* foreach fiber */

    /* OUTER PRODUCTS */

    /* flush buffer last time */
    p_flush_oprods(accum_oprod, accum_fids, avals, rankA, accum_nnz_raw, rankB,
        naccum, outv);
    naccum = 0;

  } /* foreach outer slice */
}


static void p_csf_ttmc_intl3(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{

  matrix_t const * const A = mats[csf->dim_perm[0]];
  matrix_t const * const B = mats[csf->dim_perm[2]];

  idx_t const rankA = A->J;
  idx_t const rankB = B->J;

  val_t const * const restrict vals = csf->pt[tile_id].vals;

  idx_t const * const restrict sptr = csf->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = csf->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = csf->pt[tile_id].fids[0];
  idx_t const * const restrict fids = csf->pt[tile_id].fids[1];
  idx_t const * const restrict inds = csf->pt[tile_id].fids[2];

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  int const tid = omp_get_thread_num();
  val_t * const restrict accum_nnz = (val_t *) thds[tid].scratch[0];

  /* foreach slice */
  idx_t const nslices = csf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    val_t const * const restrict av = avals + (fid * rankA);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {

      /* first entry of the fiber is used to initialize accum_nnz */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * rankB);
      for(idx_t r=0; r < rankB; ++r) {
        accum_nnz[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * rankB);
        for(idx_t r=0; r < rankB; ++r) {
          accum_nnz[r] += v * bv[r];
        }
      }
#if SPLATT_TTMC_FLOPS == 1
      #pragma omp atomic
      nflops += 2 * rankB * (fptr[f+1] - fptr[f]);
#endif

      /* accumulate outer product into tenout */
      val_t * const restrict outv = tenout + (fids[f] * rankA * rankB);
      mutex_set_lock(pool, fids[f]);
      p_twovec_outer_prod_accum(av, rankA, accum_nnz, rankB, outv);
      mutex_unset_lock(pool, fids[f]);
#if SPLATT_TTMC_FLOPS == 1
      #pragma omp atomic
      nflops += 2 * rankA * rankB;
#endif
    } /* foreach fiber */
  } /* foreach slice */
}



/**
* @brief Perform TTM on the leaf mode of a 3D CSF tensor.
*
* @param csf The input tensor.
* @param tile_id Which tile.
* @param mats Input matrices.
* @param tenout Output tensor.
* @param thds Thread structures.
*/
static void p_csf_ttmc_leaf3(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{
  assert(csf->nmodes == 3);

  matrix_t const * const A = mats[csf->dim_perm[0]];
  matrix_t const * const B = mats[csf->dim_perm[1]];

  idx_t const rankA = A->J;
  idx_t const rankB = B->J;

  val_t const * const restrict vals = csf->pt[tile_id].vals;

  idx_t const * const restrict sptr = csf->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = csf->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = csf->pt[tile_id].fids[0];
  idx_t const * const restrict fids = csf->pt[tile_id].fids[1];
  idx_t const * const restrict inds = csf->pt[tile_id].fids[2];

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  int const tid = omp_get_thread_num();

  /* nonzero accumulation */
  val_t * const accum_oprod = thds[tid].scratch[2];

  /* foreach slice */
  idx_t const nslices = csf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict av = avals + (fid * rankA);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {

      /* fill fiber with outer product */
      val_t const * const restrict bv = bvals + (fids[f] * rankB);
      p_twovec_outer_prod(av, rankA, bv, rankB, accum_oprod);

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t * const restrict outv = tenout + (inds[jj] * rankA * rankB);
        mutex_set_lock(pool, inds[jj]);
        for(idx_t r=0; r < rankA * rankB; ++r) {
          outv[r] += v * accum_oprod[r];
        }
        mutex_unset_lock(pool, inds[jj]);
      }

#if SPLATT_TTMC_FLOPS == 1
      #pragma omp atomic
      nflops += 2 * rankA * rankB * (fptr[f+1] - fptr[f]);
#endif

    } /* foreach fiber */
  }
}


static void p_csf_ttmc_leaf(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{
  /* empty tile, just return */
  val_t const * const vals = csf->pt[tile_id].vals;
  if(vals == NULL) {
    return;
  }

  idx_t const nmodes = csf->nmodes;
  if(nmodes == 3) {
    p_csf_ttmc_leaf3(csf, tile_id, mats, tenout, thds);
    return;
  }

  idx_t const mode = csf->dim_perm[nmodes-1];

  /* count the number of columns in output and each factor */
  idx_t ncols_mat[MAX_NMODES];
  ncols_mat[0] = 0; /* just get rid of a warning */
  idx_t ncols = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ncols_mat[m] = mats[csf->dim_perm[m]]->J;
    if(m != mode) {
      ncols *= mats[m]->J;
    }
  }

  /* count the number of columns at each depth -- grows from top */
  idx_t ncols_lvl[MAX_NMODES];
  ncols_lvl[0] = ncols_mat[0];
  for(idx_t m=1; m < nmodes; ++m) {
    ncols_lvl[m] = ncols_lvl[m-1] * ncols_mat[m];
  }
  ncols_lvl[nmodes-1] = ncols;

  /* allocate buffers */
  val_t * mvals[MAX_NMODES]; /* permuted factors */
  val_t * buf[MAX_NMODES];   /* accumulation buffer */
  for(idx_t m=0; m < nmodes; ++m) {
    buf[m] = calloc(ncols_lvl[m], sizeof(**buf));
    mvals[m] = mats[csf->dim_perm[m]]->vals;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) csf->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) csf->pt[tile_id].fids;

  /* foreach outer slice */
  idx_t idxstack[MAX_NMODES];
  idx_t const nslices = csf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    idxstack[0] = s;

    /* clear out stale data */
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const restrict rootrow = mvals[0] + (fid * ncols_mat[0]);
    val_t * const rootbuf = buf[0];
    for(idx_t f=0; f < ncols_mat[0]; ++f) {
      rootbuf[f] = rootrow[f];
    }

    idx_t depth = 0;

    idx_t const outer_end = fp[0][s+1];
    while(idxstack[1] < outer_end) {
      /* move down to an nnz node */
      for(; depth < nmodes-2; ++depth) {
        /* propogate buf down */
        val_t const * const restrict drow = mvals[depth+1] + \
            (fids[depth+1][idxstack[depth+1]] * ncols_mat[depth+1]);
        assert(ncols_lvl[depth+1] == ncols_lvl[depth] * ncols_mat[depth+1]);
        p_twovec_outer_prod(buf[depth], ncols_lvl[depth],
                            drow, ncols_mat[depth+1],
                            buf[depth+1]);
      }

      /* process all nonzeros [start, end) */
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      p_csf_process_fiber_lock(tenout, buf[depth],
          ncols, start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */

  /* cleanup */
  for(idx_t m=0; m < nmodes-1; ++m) {
    splatt_free(buf[m]);
  }
}




static void p_csf_ttmc_root(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = csf->nmodes;
  val_t const * const vals = csf->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_ttmc_root3(csf, tile_id, mats, tenout, thds);
    return;
  }

  idx_t const mode = csf->dim_perm[0];

  idx_t ncols_mat[MAX_NMODES];
  /* count the number of columns in output and at each depth */
  idx_t ncols = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ncols_mat[m] = mats[csf->dim_perm[m]]->J;
    if(m != mode) {
      ncols *= mats[m]->J;
    }
  }

  idx_t ncols_lvl[MAX_NMODES];
  ncols_lvl[0] = ncols;
  ncols_lvl[1] = ncols;
  /* start at 1 to skip output */
  for(idx_t m=2; m < nmodes; ++m) {
    ncols_lvl[m] = ncols_lvl[m-1] / ncols_mat[m];
  }
  ncols_lvl[nmodes-1] = ncols_mat[nmodes-1];

  val_t * mvals[MAX_NMODES]; /* permuted factors */
  val_t * buf[MAX_NMODES];   /* accumulation buffer */
  for(idx_t m=0; m < nmodes; ++m) {
    buf[m] = splatt_malloc(ncols_lvl[m] * sizeof(**buf));
    for(idx_t f=0; f < ncols_lvl[m]; ++f) {
      buf[m][f] = 0.;
    }

    mvals[m] = mats[csf->dim_perm[m]]->vals;
  }

  idx_t const nfibs = csf->pt[tile_id].nfibs[0];
  idx_t const * const * const restrict fp
      = (idx_t const * const *) csf->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) csf->pt[tile_id].fids;

  idx_t idxstack[MAX_NMODES];

  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nfibs; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    assert(fid < csf->dims[mode]);

    p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids, vals, mvals, nmodes,
        ncols_mat, ncols_lvl);

    val_t       * const restrict orow = tenout + (fid * ncols);
    val_t const * const restrict obuf = buf[0];
    for(idx_t f=0; f < ncols; ++f) {
      orow[f] += obuf[f];
    }
#if SPLATT_TTMC_FLOPS == 1
    #pragma omp atomic
    nflops += ncols;
#endif
  } /* foreach outer slice */

  /* cleanup */
  for(idx_t m=0; m < nmodes-1; ++m) {
    splatt_free(buf[m]);
  }
}



static void p_csf_ttmc_internal(
  splatt_csf const * const csf,
  idx_t const tile_id,
  matrix_t ** mats,
  val_t * const tenout,
  idx_t const mode,
  thd_info * const thds)
{
  /* empty tile, just return */
  val_t const * const vals = csf->pt[tile_id].vals;
  if(vals == NULL) {
    return;
  }

  idx_t const nmodes = csf->nmodes;
  if(nmodes == 3) {
    p_csf_ttmc_intl3(csf, tile_id, mats, tenout, thds);
    return;
  }

  /* find out which level in the tree this is */
  idx_t const outdepth = csf_mode_depth(mode, csf->dim_perm, nmodes);

  /* count the number of columns in output and each factor */
  idx_t ncols_mat[MAX_NMODES];
  ncols_mat[0] = 0;
  idx_t ncols = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ncols_mat[m] = mats[csf->dim_perm[m]]->J;
    if(m != mode) {
      ncols *= mats[m]->J;
    }
  }

  /* count the number of columns at each depth -- grows from ends */
  idx_t ncols_lvl[MAX_NMODES];
  ncols_lvl[0] = ncols_mat[0];
  for(idx_t m=1; m < nmodes-1; ++m) {
    if(m < outdepth) {
      ncols_lvl[m] = ncols_lvl[m-1] * ncols_mat[m];
    } else if(m == outdepth) {
      ncols_lvl[m] = ncols;
    } else {
      ncols_lvl[m] = ncols_lvl[m-1] / ncols_mat[m];
    }
  }
  ncols_lvl[nmodes-1] = ncols_mat[nmodes-1];

  /* allocate buffers */
  val_t * mvals[MAX_NMODES]; /* permuted factors */
  val_t * buf[MAX_NMODES];   /* accumulation buffer */
  for(idx_t m=0; m < nmodes; ++m) {
    buf[m] = calloc(ncols_lvl[m], sizeof(**buf));
    mvals[m] = mats[csf->dim_perm[m]]->vals;
  }


  /* extract tensor structures */
  idx_t const * const * const restrict fp
      = (idx_t const * const *) csf->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) csf->pt[tile_id].fids;


  /* foreach outer slice */
  idx_t idxstack[MAX_NMODES];
  idx_t const nslices = csf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    /* push outer slice and fill stack */
    idxstack[0] = s;
    for(idx_t m=1; m <= outdepth; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* fill first buf */
    val_t const * const restrict rootrow = mvals[0] + (fid * ncols_mat[0]);
    for(idx_t f=0; f < ncols_mat[0]; ++f) {
      buf[0][f] = rootrow[f];
    }

    /* process entire subtree */
    idx_t depth = 0;
    while(idxstack[1] < fp[0][s+1]) {
      /* propagate values down to outdepth-1 */
      for(; depth < outdepth-1; ++depth) {
        val_t const * const restrict drow = mvals[depth+1] + \
            (fids[depth+1][idxstack[depth+1]] * ncols_mat[depth+1]);
        assert(ncols_lvl[depth+1] == ncols_lvl[depth] * ncols_mat[depth+1]);
        p_twovec_outer_prod(buf[depth], ncols_lvl[depth],
                            drow, ncols_mat[depth+1],
                            buf[depth+1]);
#if SPLATT_TTMC_FLOPS == 1
        #pragma omp atomic
        nflops += 2 * (ncols_lvl[depth] * ncols_mat[depth+1]);
#endif
      }
      ++depth;
      assert(depth == outdepth);

      /* propagate value up to buf[outdepth] -- actually only
       * ncols_lvl[outdepth+1] values! */
      p_propagate_up(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
          fp, fids, vals, mvals, nmodes, ncols_mat, ncols_lvl);


      /* update output row */
      idx_t const noderow = fids[outdepth][idxstack[outdepth]];
      val_t * const restrict outv = tenout + (noderow * ncols);
      assert(ncols == ncols_lvl[outdepth-1] * ncols_lvl[outdepth+1]);
      mutex_set_lock(pool, noderow);
      p_twovec_outer_prod_accum(buf[outdepth-1], ncols_lvl[outdepth-1],
                                buf[outdepth], ncols_lvl[outdepth+1],
                                outv);
      mutex_unset_lock(pool, noderow);
#if SPLATT_TTMC_FLOPS == 1
      #pragma omp atomic
      nflops += 2 * (ncols_lvl[depth-1] * ncols_lvl[depth+1]);
#endif
      /* clear buffer -- hacked to be outdepth+1 */
      for(idx_t f=0; f < ncols_lvl[outdepth+1]; ++f) {
        buf[outdepth][f] = 0.;
      }

      /* backtrack to next unfinished node */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end foreach outer slice */

  /* cleanup */
  for(idx_t m=0; m < nmodes-1; ++m) {
    splatt_free(buf[m]);
  }
}





static inline void p_root_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  timer_start(&thds[omp_get_thread_num()].ttime);
  /* tile id */
  idx_t tid = 0;
  switch(tensor->which_tile) {
  case SPLATT_NOTILE:
    p_csf_ttmc_root(tensor, 0, mats, tenout, thds);
    break;

  /* XXX */
  default:
    fprintf(stderr, "SPLATT: TTM does not support tiling yet.\n");
    exit(1);
    break;
  }
  timer_stop(&thds[omp_get_thread_num()].ttime);

}



static void p_intl_decide(
    splatt_csf const * const csf,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{

  idx_t const nmodes = csf->nmodes;

  timer_start(&thds[omp_get_thread_num()].ttime);
  /* tile id */
  idx_t tid = 0;
  switch(csf->which_tile) {
  case SPLATT_NOTILE:
    p_csf_ttmc_internal(csf, 0, mats, tenout, mode, thds);
    break;

  /* XXX */
  default:
    fprintf(stderr, "SPLATT: TTM does not support tiling yet.\n");
    exit(1);
    break;
  }
  timer_stop(&thds[omp_get_thread_num()].ttime);
}



static void p_leaf_decide(
    splatt_csf const * const csf,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = csf->nmodes;
  idx_t const depth = nmodes - 1;

  timer_start(&thds[omp_get_thread_num()].ttime);

  /* tile id */
  idx_t tid = 0;
  switch(csf->which_tile) {
  case SPLATT_NOTILE:
    p_csf_ttmc_leaf(csf, 0, mats, tenout, thds);
    break;

  /* XXX */
  default:
    fprintf(stderr, "SPLATT: TTM does not support tiling yet.\n");
    exit(1);
    break;
  }
  timer_stop(&thds[omp_get_thread_num()].ttime);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


void ttmc_csf(
    splatt_csf const * const tensors,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  nflops = 0;

  /* clear out stale results */
  p_clear_tenout(tenout, mats, tensors->nmodes, mode, tensors->dims);

  splatt_omp_set_num_threads(opts[SPLATT_OPTION_NTHREADS]);
  if(pool == NULL) {
    pool = mutex_alloc();
  }

  idx_t const nrows = tensors[0].dims[mode];
  idx_t ncols = 1;
  for(idx_t m=0; m < tensors[0].nmodes; ++m) {
    if(m != mode) {
      ncols *= mats[m]->J;
    }
  }

  val_t * output_bufs[splatt_omp_get_max_threads()];

  bool privatized = false;
  if(false && tensors[0].dims[mode] < 1000) {
    privatized = true;
  }

  sp_timer_t ttmc_time;
  timer_fstart(&ttmc_time);

  #pragma omp parallel
  {
    val_t * ttmc_output = tenout;

    if(privatized) {
      size_t const nbytes = nrows * ncols * sizeof(**output_bufs);
      int const tid = splatt_omp_get_thread_num();
      output_bufs[tid] = splatt_malloc(nbytes);
      memset(output_bufs[tid], 0, nbytes);

      /* set thread-local output */
      ttmc_output = output_bufs[tid];
    }

    idx_t nmodes = tensors[0].nmodes;
    /* find out which level in the tree this is */
    idx_t outdepth = MAX_NMODES;

    /* choose which TTM function to use */
    splatt_csf_type which = opts[SPLATT_OPTION_CSF_ALLOC];
    switch(which) {

    case SPLATT_CSF_ONEMODE:
      outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
      if(outdepth == 0) {
        p_root_decide(tensors+0, mats, ttmc_output, mode, thds, opts);
      } else if(outdepth == nmodes - 1) {
        p_leaf_decide(tensors+0, mats, ttmc_output, mode, thds, opts);
      } else {
        p_intl_decide(tensors+0, mats, ttmc_output, mode, thds, opts);
      }
      break;


    case SPLATT_CSF_TWOMODE:
      /* longest mode handled via second tensor's root */
      if(mode == tensors[0].dim_perm[nmodes-1]) {
        p_root_decide(tensors+1, mats, ttmc_output, mode, thds, opts);
      /* root and internal modes are handled via first tensor */
      } else {
        outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
        if(outdepth == 0) {
          p_root_decide(tensors+0, mats, ttmc_output, mode, thds, opts);
        } else {
          p_intl_decide(tensors+0, mats, ttmc_output, mode, thds, opts);
        }
      }
      break;

    case SPLATT_CSF_ALLMODE:
      p_root_decide(tensors+mode, mats, ttmc_output, mode, thds, opts);
      break;

    default:
      fprintf(stderr, "SPLATT: only SPLATT_CSF_ALLMODE supported for TTM.\n");
      exit(1);
    }

    if(privatized) {
      #pragma omp barrier

      /* Perform a reduction on the thread-private output buffers. */
      #pragma omp for schedule(static)
      for(idx_t i=0; i < nrows; ++i) {
        val_t * const restrict tenout_slice = tenout + (i*ncols);

        for(int t=0; t < splatt_omp_get_num_threads(); ++t) {
          val_t const * const restrict thread_slice = output_bufs[t]+(i*ncols);
          for(idx_t j=0; j < ncols; ++j) {
            tenout_slice[j] += thread_slice[j];
          }
        }
      }

      /* free thread memory */
      splatt_free(output_bufs[splatt_omp_get_thread_num()]);
    } /* if parallelized */
  } /* end omp parallel */

  timer_stop(&ttmc_time);

#if SPLATT_TTMC_FLOPS == 1
  printf("    TTMc: %0.3fs (%0.3f GFLOPS)\n", ttmc_time.seconds,
      1e-9 * (double) nflops / ttmc_time.seconds);
#else
  printf("    TTMc: %0.3fs\n", ttmc_time.seconds);
#endif
}



void ttmc_stream(
    sptensor_t const * const tt,
    matrix_t ** mats,
    val_t * const tenout,
    idx_t const mode,
    double const * const opts)
{
  idx_t const nmodes = tt->nmodes;

  if(pool == NULL) {
    pool = mutex_alloc();
  }

  /* clear out stale results */
  p_clear_tenout(tenout, mats, nmodes, mode, tt->dims);

  #pragma omp parallel
  {
    val_t const * const restrict vals = tt->vals;
    val_t * mvals[MAX_NMODES];
    idx_t nfactors[MAX_NMODES];

    idx_t total_cols = 1;

    /* number of columns accumulated by the m-th mode (counting backwards) */
    idx_t ncols[MAX_NMODES+1];
    ncols[nmodes] = 1;
    val_t * buffers[MAX_NMODES];

    for(idx_t m=nmodes; m-- != 0; ) {
      nfactors[m] = mats[m]->J;
      mvals[m] = mats[m]->vals;

      /* allocate buffer */
      if(m != mode) {
        total_cols *= nfactors[m];
        ncols[m] = ncols[m+1] * nfactors[m];
        buffers[m] = splatt_malloc(ncols[m] * sizeof(**buffers));
      } else {
        ncols[m] = ncols[m+1];
        buffers[m] = NULL;
      }
    }
    assert(total_cols == ncols[0]);

    /* the last mode we accumulate with */
    idx_t const first_mode = (mode == 0) ? 1 : 0;
    idx_t const last_mode  = (mode == nmodes-1) ? nmodes-2 : nmodes-1;

    #pragma omp for schedule(static)
    for(idx_t n=0; n < tt->nnz; ++n) {
      val_t * const restrict outrow = tenout + (tt->ind[mode][n] * total_cols);

      /* initialize buffer with nonzero value */
      val_t * curr_buff = buffers[last_mode];
      idx_t buff_size = nfactors[last_mode];
      val_t const * const restrict last_row =
          mvals[last_mode] + (tt->ind[last_mode][n] * nfactors[last_mode]);
      for(idx_t f=0; f < nfactors[last_mode]; ++f) {
        curr_buff[f] = vals[n] * last_row[f];
      }

      /* now do nmodes-1 kronecker products */
      for(idx_t m=last_mode; m-- != 0; ) {
        if(m == mode) {
          continue;
        }

        if(m != first_mode) {
          /* outer product */
          p_twovec_outer_prod(
              mvals[m] + (tt->ind[m][n] * nfactors[m]), nfactors[m],
              curr_buff, buff_size,
              buffers[m]);
        } else {
          mutex_set_lock(pool, tt->ind[mode][n]);
          /* accumulate into output on the first mode (last to be processed) */
          p_twovec_outer_prod_accum(
              mvals[m] + (tt->ind[m][n] * nfactors[m]), nfactors[m],
              curr_buff, buff_size,
              outrow);
          mutex_unset_lock(pool, tt->ind[mode][n]);
        }

        /* next buffer */
        curr_buff = buffers[m];
        buff_size *= nfactors[m];
      }
    }

    for(idx_t m=0; m < nmodes; ++m) {
      splatt_free(buffers[m]);
    }
  } /* end omp parallel */
}



void ttmc_largest_outer(
    splatt_csf const * const tensors,
    idx_t * const outer_sizes,
    double const * const opts)
{
  idx_t const ntensors = csf_ntensors(tensors, opts);
  idx_t const nmodes = tensors->nmodes;

  memset(outer_sizes, 0, nmodes * sizeof(*outer_sizes));

  for(idx_t t=0; t < ntensors; ++t) {
    splatt_csf const * const csf = &(tensors[t]);
    idx_t const ntiles = csf->ntiles;

    for(idx_t tile=0; tile < ntiles; ++tile) {
      /* don't count mode above nnz; they accumulate instead of oprod */
      for(idx_t m=0; m < nmodes-2; ++m) {
        idx_t const madj = csf->dim_perm[m];
        idx_t const * const fptr = csf->pt[tile].fptr[m];
        idx_t const nfibs = csf->pt[tile].nfibs[m];

        for(idx_t f=0; f < nfibs; ++f) {
          outer_sizes[madj] = SS_MAX(outer_sizes[madj], fptr[f+1] - fptr[f]);
        }
      }
    }
  }
}


idx_t tenout_dim(
    idx_t const nmodes,
    idx_t const * const nfactors,
    idx_t const * const dims)
{
  idx_t maxdim = 0;

  /* compute the size for each mode and maintain max */
  for(idx_t m=0; m < nmodes; ++m) {
    idx_t const nrows = dims[m];
    idx_t const ncols = p_ttmc_outncols(nfactors, nmodes, m);
    maxdim = SS_MAX(maxdim, nrows * ncols);
  }

  return maxdim;
}


void ttmc_fill_flop_tbl(
    sptensor_t * const tt,
    idx_t const * const nfactors,
    size_t table[MAX_NMODES][MAX_NMODES])
{
  /* just assume no tiling... */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

  /* flops if we used just CSF-1 or CSF-A */
  size_t csf1[MAX_NMODES];
  size_t csf2[MAX_NMODES];
  size_t csfa[MAX_NMODES];

  idx_t const smallest_mode = argmin_elem(tt->dims, tt->nmodes);
  idx_t const largest_mode = argmax_elem(tt->dims, tt->nmodes);

  /* foreach CSF rep */
  for(idx_t i=0; i < tt->nmodes; ++i) {
    printf("MODE-%"SPLATT_PF_IDX":  ", i);

    splatt_csf csf;
    csf_alloc_mode(tt, CSF_SORTED_SMALLFIRST_MINUSONE, i, &csf, opts);

    /* foreach mode of computation */
    for(idx_t j=0; j < tt->nmodes; ++j) {

      size_t const flops = ttmc_csf_count_flops(&csf, j, nfactors);
      /* store result */
      table[i][j] = flops;
      printf("%0.3e  ", (double)flops);

      if(i == smallest_mode) {
        csf1[j] = flops;
        if(j != largest_mode) {
          csf2[j] = flops;
        }
      }
      if(i == j) {
        csfa[i] = flops;

        /* csf-2 uses special leaf mode */
        if(i == largest_mode) {
          csf2[j] = flops;
        }
      }
    } /* end foreach mode of computation */

    size_t total = 0;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      total += table[i][m];
    }
    printf(" = %0.3e\n", (double)total);

    csf_free_mode(&csf);
  }
  splatt_free_opts(opts);
  printf("\n");



  /* print stats for each allocation scheme */

  size_t total;

  /* csf-1 and csf-a */
  total = 0;
  printf("CSF-1:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("%0.3e  ", (double)csf1[m]);
    total += csf1[m];
  }
  printf(" = %0.3e\n", (double)total);
  total = 0;
  printf("CSF-2:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("%0.3e  ", (double)csf2[m]);
    total += csf2[m];
  }
  printf(" = %0.3e\n", (double)total);

  total = 0;
  printf("CSF-A:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("%0.3e  ", (double)csfa[m]);
    total += csfa[m];
  }
  printf(" = %0.3e\n", (double)total);

  bool mode_used[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mode_used[m] = false;
  }

  /* handpick best modes */
  printf("CUSTM:  ");
  total = 0;
  /* foreach mode */
  for(idx_t j=0; j < tt->nmodes; ++j) {
    size_t best = 0;
    /* foreach csf */
    for(idx_t i=0; i < tt->nmodes; ++i) {
      if(table[i][j] <= table[best][j]) {
        best = i;
      }
    }

    mode_used[best] = true;

    total += table[best][j];
    printf("%0.3e  ", (double) table[best][j]);
  }
  printf(" = %0.3e\n", (double) total);



  /* coordinate form */
  total = 0;
  printf("COORD:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    size_t const coord_flops = ttmc_coord_count_flops(tt, m, nfactors);
    printf("%0.3e  ", (double)coord_flops);
    total += coord_flops;
  }
  printf(" = %0.3e\n", (double)total);
  printf("\n");

  /* print CSF needed */
  printf("CUSTOM MODES:");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(mode_used[m]) {
      printf(" %"SPLATT_PF_IDX, m);
    }
  }
  printf("\n");

}


size_t ttmc_csf_count_flops(
    splatt_csf const * const csf,
    idx_t const mode,
    idx_t const * const nfactors)
{
  idx_t const depth = csf_mode_depth(mode, csf->dim_perm, csf->nmodes);

  size_t flops = 0;

  /* foreach tile */
  for(idx_t tile=0; tile < csf->ntiles; ++tile) {
    size_t out_size = (size_t)nfactors[csf->dim_perm[0]];

    /* move down tree (no 2 needed as it is assignment, not accumulation) */
    for(idx_t d=1; d < depth; ++d) {
      out_size *= (size_t) nfactors[csf->dim_perm[d]];
      flops += (size_t) csf->pt[tile].nfibs[d] * out_size;
    }

    out_size = 1;

    /* move up tree */
    /* nmodes -> depth (exclusive) */
    for(idx_t d=csf->nmodes; d-- != depth+1; ) {
      out_size *= (size_t) nfactors[csf->dim_perm[d]];
      flops += 2 * (size_t) csf->pt[tile].nfibs[d] * out_size;
    }

    /* final join if internal/leaf mode */
    if(depth > 0) {
      out_size = (size_t) p_ttmc_outncols(nfactors, csf->nmodes, mode);
      flops += 2 * (size_t)csf->pt[tile].nfibs[depth] * out_size;
    }
  } /* end foreach tile */

  return flops;
}



size_t ttmc_coord_count_flops(
    sptensor_t const * const tt,
    idx_t const mode,
    idx_t const * const nfactors)
{
  /* flops to grow kronecker products */
  size_t accum_flops = 0;
  size_t ncols = nfactors[0];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(m != mode) {
      accum_flops += (size_t) tt->nnz * ncols;
      ncols *= (size_t) nfactors[m];
    }
  }

  /* add actual addition to output tensor */
  accum_flops += 2 * (size_t) tt->nnz \
     *  (size_t) p_ttmc_outncols(nfactors, tt->nmodes, mode);

  return accum_flops;
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



void ttmc_compute_ncols(
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



void permute_core(
    splatt_csf const * const tensors,
    val_t * const core,
    idx_t const * const nfactors,
    double const * const opts)
{
  idx_t const nmodes = tensors->nmodes;
  idx_t ncols[MAX_NMODES+1];
  ttmc_compute_ncols(nfactors, nmodes, ncols);

  idx_t perm[MAX_NMODES];
  p_fill_core_perm(tensors, perm, opts);

  val_t * newcore = splatt_malloc(ncols[nmodes] * sizeof(*newcore));

  /* translate each entry in the core tensor individually */
  #pragma omp parallel for
  for(idx_t x=0; x < ncols[nmodes]; ++x) {
    idx_t ind[MAX_NMODES];
    /* translate x into ind, respecting the natural ordering of modes */
    idx_t id = x;
    for(idx_t m=nmodes; m-- != 0; ){
      ind[m] = id % nfactors[m];
      id /= nfactors[m];
    }

    /* translate ind to an index into core, accounting for permutation */
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
  par_memcpy(core, newcore, ncols[nmodes] * sizeof(*core));
  splatt_free(newcore);
}



thd_info * ttmc_alloc_thds(
    idx_t const nthreads,
    splatt_csf const * const tensors,
    idx_t const * const nfactors,
    double const * const opts)
{
  idx_t const nmodes = tensors->nmodes;

  /* find # columns for each TTMc and output core */
  idx_t ncols[MAX_NMODES+1];
  ttmc_compute_ncols(nfactors, nmodes, ncols);
  idx_t const maxcols = ncols[argmax_elem(ncols, nmodes)];

  idx_t const maxfactor = nfactors[argmax_elem(nfactors, nmodes)];

  thd_info * thds = thd_init(nthreads, 3,
    /* nnz accumulation & buffers */
    (TTMC_BUFROWS * maxcols * sizeof(val_t)),
    /* fids */
    (TTMC_BUFROWS * sizeof(idx_t)) ,
    /* actual rows corresponding to fids */
    (maxfactor * TTMC_BUFROWS * sizeof(val_t)));

  return thds;
}



