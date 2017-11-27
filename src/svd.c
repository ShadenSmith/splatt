

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "svd.h"
#include "splatt_lapack.h"
#include "matrix.h"
#include "timer.h"
#include "util.h"
#include "io.h"

#include <math.h>


/* Minimum #row / #vecs ratio to use Lanczos */
double const MIN_ROW_RATIO_LANCZOS = 10.;


/**
* @brief The number of extra vectors (and Lanczos iterations) to compute, as
*        a multiplier of the given rank (i.e., 2x the number of singular
*        vectors).
*/
idx_t const LANCZOS_EXTRA_VECS = 2;


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/





/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/



void left_singulars(
    matrix_t const * const A,
    matrix_t       * const left_singular_matrix,
    idx_t const nvecs,
    svd_ws * const ws)
{
	timer_start(&timers[TIMER_SVD]);

  /* decide which SVD to use */
  double const ratio = (double) A->I / (double) nvecs;
  if(ratio >= MIN_ROW_RATIO_LANCZOS) {
    left_singulars_lanczos(A, left_singular_matrix, nvecs, ws);
  } else {
    left_singulars_lapack(A, left_singular_matrix, nvecs, ws);
  }

	timer_stop(&timers[TIMER_SVD]);
}


void left_singulars_lapack(
    matrix_t const * const A,
    matrix_t       * const left_singular_matrix,
    idx_t const nvecs,
    svd_ws * const ws)
{
  char jobz = 'S';

  /* actually pass in A^T */
  splatt_blas_int M = A->J;
  splatt_blas_int N = A->I;
  splatt_blas_int LDA = M;

  val_t * inmat = splatt_malloc(M * N * sizeof(*inmat));
  par_memcpy(inmat, A->vals, M * N * sizeof(*inmat));

  val_t * S = splatt_malloc(SS_MIN(M,N) * sizeof(*S));
  par_memset(S, 0, SS_MIN(M,N) * sizeof(*S));

  /* NOTE: change these if we switch to jobz=O */
  splatt_blas_int LDU = M;
  splatt_blas_int LDVt = SS_MIN(M,N);

  val_t * U  = splatt_malloc(LDU * SS_MIN(M,N) * sizeof(*U));
  val_t * Vt = splatt_malloc(LDVt * N * sizeof(*Vt));

  val_t work_size;
  splatt_blas_int lwork = -1;
  splatt_blas_int * iwork = splatt_malloc(8 * SS_MIN(M,N) * sizeof(*iwork));
  splatt_blas_int info = 0;

  /* query */
  SPLATT_BLAS(gesdd)(
      &jobz,
      &M, &N,
      inmat, &LDA,
      S,
      U, &LDU,
      Vt, &LDVt,
      &work_size, &lwork,
      iwork, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGESDD returned %d\n", info);
  }

  lwork = work_size;
  val_t * workspace = splatt_malloc(lwork * sizeof(*workspace));

  /* do the SVD */
  SPLATT_BLAS(gesdd)(
      &jobz,
      &M, &N,
      inmat, &LDA,
      S,
      U, &LDU,
      Vt, &LDVt,
      workspace, &lwork,
      iwork, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGESDD returned %d\n", info);
  }


  /* copy matrix Vt to outmat */
  idx_t const rank = left_singular_matrix->J;
  idx_t const effective_rank = SS_MIN(rank, (idx_t) N);
  val_t * const restrict outmat = left_singular_matrix->vals;
  par_memset(outmat, 0, rank * A->I * sizeof(*outmat));
  #pragma omp parallel
  for(idx_t c=0; c < effective_rank; ++c) {
    #pragma omp for schedule(static)
    for(idx_t r=0; r < A->I; ++r) {
      outmat[c + (r*rank)] = Vt[c + (r*LDVt)];
    }
  }

  splatt_free(workspace);
  splatt_free(iwork);
  splatt_free(inmat);
  splatt_free(S);
  splatt_free(Vt);
  splatt_free(U);
}


void left_singulars_lanczos(
    matrix_t const * const A,
    matrix_t       * const left_singular_matrix,
    idx_t const nvecs,
    svd_ws * const ws)
{
  left_singular_matrix->J = nvecs;

  idx_t const lanczos_nvecs = SS_MIN(A->J, nvecs * LANCZOS_EXTRA_VECS);

  /* Compute the bidiagonalization and right orthogonal matrix */
  lanczos_onesided_bidiag(A, lanczos_nvecs, ws);

  char uplo = 'U';
  int n = lanczos_nvecs;

  int ncvt = A->J;
  int nru  = 0;
  int ncc  = 0;

  /* switch to row-major -- same as transpose & same dimensions */
  mat_transpose(ws->Q, ws->Qt);
  ws->Qt->I = ws->Q->I;
  ws->Qt->J = ws->Q->J;

  /* LAPACK_BDSQR computes the SVD of the bidiagonalized A. and also the
   * right singular vectors */
  {
    val_t * VT = ws->Qt->vals;
    val_t * U = NULL;
    val_t * C = NULL;

    int ldu = SS_MAX(1, nru);
    int ldvt = 1;
    if(ncvt > 0) {
      ldvt = SS_MAX(1, lanczos_nvecs);
    }
    int ldc = 1;
    if(ncc > 0) {
      ldc = SS_MAX(1, lanczos_nvecs);
    }

    /* bi-diagonal SVD */
    int info = 0;
    SPLATT_BLAS(bdsqr)(
        &uplo, &n,
        &ncvt,
        &nru,
        &ncc,
        ws->alphas, ws->betas,
        VT, &ldvt,
        U, &ldu,
        C, &ldc,
        ws->work, &info);
    if(info != 0) {
      printf("DBDSQR returned %d\n", info);
    }
  } /* bidiagonal SVD */

  /* We have singular values and right singular vectors. Now to compute
   *  U = A * (V^T * S^-1), or U^T = (V * S^-1) * A^T  when column-major */

  /* scale Vt by S^-1 */
  #pragma omp parallel for schedule(static)
  for(idx_t i=0; i < ws->Qt->I; ++i) {
    val_t * const restrict row = ws->Qt->vals + (i * ws->Qt->J);
    val_t const * const restrict svals = ws->alphas;
    for(idx_t j=0; j < nvecs; ++j) {
      row[j] /= svals[j];
    }
  }

  /* DGEMM to get left singular vectors. Note that we switch back to 'nvecs'
   * now. */
  {
    val_t alpha = 1.;
    val_t beta  = 0.;
    int m = nvecs;
    int n = A->I;
    int k = A->J;

    int lda = ws->Qt->J; /* is >= nvecs */
    int ldb = k;
    int ldc = m;

    char transA = 'N';
    char transB = 'N';

    SPLATT_BLAS(gemm)(&transA, &transB,
               &m, &n, &k,
               &alpha,
               ws->Qt->vals, &lda,
               A->vals, &ldb,
               &beta,
               left_singular_matrix->vals, &ldc);
  }

}



void lanczos_bidiag(
    matrix_t const * const A,
    idx_t const rank,
    svd_ws * const ws)
{
  idx_t const nrows = A->I;
  idx_t const ncols = A->J;

  /* allocate P if necessary */
  if(ws->P != NULL) {
    if(ws->P->I != nrows || ws->P->J != rank) {
      mat_free(ws->P);
      ws->P = NULL;
    }
  }
  if(ws->P == NULL) {
    ws->P = mat_alloc(nrows, rank);
    ws->P->rowmajor = 0;
  }

  /* ensure Q is the correct dimension */
  ws->Q->I = ncols;
  ws->Q->J = rank;

  /* randomly initialize first column of Q and then normalize */
  fill_rand(ws->Q->vals, ncols);
  vec_normalize(ws->Q->vals, ncols);

  /* foreach column */
  for(idx_t col=0; col < rank; ++col) {
    val_t * const Pv = ws->P->vals + (col * nrows);
    val_t * const Qv = ws->Q->vals + (col * ncols);

    /* P(:,j) = A * Q(:,j) */
    mat_vec(A, Qv, Pv);

    /* orthogonalize P(:,j) against previous columns */
    mat_col_orth(ws->P, col);

    /* normalize P(:,j) */
    ws->alphas[col] = vec_normalize(Pv, nrows);

    /* compute Q(:,j+1) */
    if(col+1 < rank) {
      mat_transpose_vec(A, Pv, Qv + ncols);
      mat_col_orth(ws->Q, col+1);
      ws->betas[col] = vec_normalize(Qv + ncols, ncols);
    }
  } /* outer loop */
}


void lanczos_onesided_bidiag(
    matrix_t const * const A,
    idx_t const rank,
    svd_ws * const ws)
{
  idx_t const nrows = A->I;
  idx_t const ncols = A->J;

  /* grab the pointers to make swapping local */
  matrix_t * p0 = ws->p0;
  matrix_t * p1 = ws->p1;

  /* ensure Q is the correct dimension */
  ws->Q->I = ncols;
  ws->Q->J = rank;
  p0->I = nrows;
  p1->I = nrows;

  /* randomly initialize first column of Q and then normalize */
  fill_rand(ws->Q->vals, ncols);
  vec_normalize(ws->Q->vals, ncols);

  for(idx_t col=0; col < rank; ++col) {
    val_t * const restrict Qv = ws->Q->vals + (col * ncols);
    val_t * const restrict p0v = p0->vals;
    val_t * const restrict p1v = p1->vals;

    /* p1 = A * Q(:,j) - (beta[col-1] * p0) */
    mat_vec(A, Qv, p1v);
    if(col > 0) {
      val_t const beta = ws->betas[col-1];
      #pragma omp parallel for schedule(static)
      for(idx_t i=0; i < nrows; ++i) {
        p1v[i] -= beta * p0v[i];
      }
    }

    /* normalize P(:,j) */
    ws->alphas[col] = vec_normalize(p1v, nrows);

    /* compute Q(:,j+1) */
    if(col+1 < rank) {
      mat_transpose_vec(A, p1v, Qv + ncols);
      mat_col_orth(ws->Q, col+1);
      ws->betas[col] = vec_normalize(Qv + ncols, ncols);
    }

    /* swap p0 and p1 */
    matrix_t * swp = p0;
    p0 = p1;
    p1 = swp;
  }
}


void alloc_svd_ws(
    svd_ws * const ws,
    idx_t const nmats,
    idx_t const * const nrows,
    idx_t const * const ncolumns,
    idx_t const * const ranks)
{
  /* maximum bidiagonalization size */
  idx_t const max_nrow = nrows[argmax_elem(nrows, nmats)];
  idx_t const max_ncol = ncolumns[argmax_elem(ncolumns, nmats)];
  idx_t const max_rank = ranks[argmax_elem(ranks, nmats)];

  /* allocate space for bidiagonalization */
  idx_t const max_bi_rank = max_rank * LANCZOS_EXTRA_VECS;
  ws->alphas = splatt_malloc(max_bi_rank * sizeof(*ws->alphas));
  ws->betas  = splatt_malloc((max_bi_rank-1) * sizeof(*ws->betas));
  ws->p0 = mat_alloc(max_nrow, 1);
  ws->p1 = mat_alloc(max_nrow, 1);

  /* Q is (ncols x Lanczos rank) */
  idx_t max_bi_mode = 0;
  idx_t max_bi_elems = 0;
  for(idx_t m=0; m < nmats; ++m) {
    idx_t elems = ncolumns[m] \
        * SS_MIN(ncolumns[m], ranks[m] * LANCZOS_EXTRA_VECS);
    if(elems > max_bi_elems) {
      max_bi_elems = elems;
      max_bi_mode = m;
    }
  }
  ws->Q = mat_alloc(ncolumns[max_bi_mode],
                    SS_MIN(ncolumns[max_bi_mode],
                        ranks[max_bi_mode] * LANCZOS_EXTRA_VECS));

  ws->Qt = mat_alloc(ws->Q->I, ws->Q->J);
  ws->Q->rowmajor = 0;

  /* only used if full bidiagonalization */
  ws->P = NULL;

  ws->work = splatt_malloc(4 * max_bi_rank * sizeof(*ws->work));
}


void free_svd_ws(
    svd_ws * const ws)
{
  splatt_free(ws->alphas);
  splatt_free(ws->betas);
  splatt_free(ws->work);

  if(ws->P != NULL) {
    mat_free(ws->P);
  }
  mat_free(ws->Q);
  mat_free(ws->Qt);
  mat_free(ws->p0);
  mat_free(ws->p1);
}

