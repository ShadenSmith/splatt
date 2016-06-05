

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "svd.h"
#include "matrix.h"
#include "timer.h"
#include "util.h"

#include <math.h>



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Compute the optimal workspace size for LAPACK_SVD.
*
* @param nrows The number of rows.
* @param ncols The number of columns.
*
* @return The optimal workspace size.
*/
static int p_optimal_svd_work_size(
    idx_t const nrows,
    idx_t const ncols)
{
  char jobz = 'S';

  /* actually pass in A^T */
  int M = ncols;
  int N = nrows;
  int LDA = M;

  val_t * S = NULL; //malloc(SS_MIN(M,N) * sizeof(*S));

  /* NOTE: change these if we switch to jobz=O */
  int LDU = M;
  int LDVt = SS_MIN(M,N);

  val_t * U = NULL; //malloc(LDU * SS_MIN(M,N) * sizeof(*U));
  val_t * Vt = NULL; //malloc(LDVt * N * sizeof(*Vt));

  val_t work_size = 0;
  int * iwork = NULL; //malloc(8 * SS_MIN(M,N) * sizeof(*iwork));
  int info = 0;

  val_t * A = NULL;

  /* query */
  int lwork = -1;
  LAPACK_SVD(
      &jobz,
      &M, &N,
      A, &LDA,
      S,
      U, &LDU,
      Vt, &LDVt,
      &work_size, &lwork,
      iwork, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGESDD returned %d\n", info);
  }

  return (int) work_size;
}





/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/



void ffast_left_singulars(
    matrix_t const * const A,
    matrix_t       * const outmat,
    idx_t const nvecs,
    svd_ws * const ws)
{
	timer_start(&timers[TIMER_SVD]);

  /* compute the bidiagonalization */
  lanczos_onesided_bidiag(A, nvecs, ws);

  char uplo = 'U';
  int n = nvecs;

  int ncvt = A->J;
  int nru  = 0;
  int ncc  = 0;

  val_t * VT = ws->S;
  val_t * U = NULL;
  val_t * C = NULL;

  int ldu = SS_MAX(1, nru);
  int ldvt = 1;
  if(ncvt > 0) {
    ldvt = SS_MAX(1, nvecs);
  }
  int ldc = 1;
  if(ncc > 0) {
    ldc = SS_MAX(1, nvecs);
  }

  val_t * work = splatt_malloc(4 * n * sizeof(*work));

  int info = 0;
  dbdsqr_(&uplo, &n,
          &ncvt,
          &nru,
          &ncc,
          ws->alphas, ws->betas,
          VT, &ldvt,
          U, &ldu,
          C, &ldc,
          work, &info);
  if(info != 0) {
    printf("DBDSQR returned %d\n", info);
  }

  splatt_free(work);

	timer_stop(&timers[TIMER_SVD]);
}



void fast_left_singulars(
    matrix_t const * const A,
    idx_t const rank,
    svd_ws * const ws)
{
	timer_start(&timers[TIMER_SVD]);

  /* compute the bidiagonalization */
  lanczos_bidiag(A, rank, ws);

  char uplo = 'U';
  int n = rank;

  /* only compute left singular vectors */
  int ncvt = 0;
  int nru = A->I;
  int ncc = 0;

  int ldvt = 1;
  int ldu = nru;
  int ldc = 1;
  int info;

  val_t * work = splatt_malloc(4 * n * sizeof(*work));

  dbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, ws->alphas, ws->betas, NULL, &ldu,
      ws->P->vals, &ldu, NULL, &ldc, work, &info);
  if(info) {
    printf("DBDSQR returned %d\n", info);
  }

  splatt_free(work);

	timer_stop(&timers[TIMER_SVD]);
}



void lanczos_bidiag(
    matrix_t const * const A,
    idx_t const rank,
    svd_ws * const ws)
{
  idx_t const nrows = A->I;
  idx_t const ncols = A->J;

  /* alloc if first time */
  if(ws->P == NULL) {
    ws->P = mat_alloc(nrows, rank);
    ws->Q = mat_alloc(ncols, rank);
    ws->P->rowmajor = 0;
    ws->Q->rowmajor = 0;

    ws->betas = splatt_malloc((rank-1) * sizeof(*ws->betas));
    ws->alphas = splatt_malloc(rank * sizeof(*ws->betas));
  }

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

  matrix_t * p0 = mat_alloc(nrows, 1);
  matrix_t * p1 = mat_alloc(nrows, 1);

  /* alloc if first time */
  if(ws->Q == NULL) {
    ws->Q = mat_alloc(ncols, rank);
    ws->Q->rowmajor = 0;
    ws->alphas = splatt_malloc(rank * sizeof(*ws->betas));
    ws->betas = splatt_malloc((rank-1) * sizeof(*ws->betas));
  }

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

  mat_free(p0);
  mat_free(p1);
}




void left_singulars(
    val_t const * const inmat,
    val_t * outmat,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const rank,
    svd_ws * const ws)
{
	timer_start(&timers[TIMER_SVD]);

  char jobz = 'S';

  /* actually pass in A^T */
  int M = ncols;
  int N = nrows;
  int LDA = M;
  int LDU = M;
  int LDVt = SS_MIN(M,N);

  int info = 0;

  memcpy(ws->A, inmat, M * N * sizeof(*(ws->A)));

  /* do the SVD */
  LAPACK_SVD(
      &jobz,
      &M, &N,
      ws->A, &LDA,
      ws->S,
      ws->U, &LDU,
      ws->Vt, &LDVt,
      ws->workspace, &(ws->lwork),
      ws->iwork, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DGESDD returned %d\n", info);
  }


  /* copy matrix Vt to outmat */
  val_t const * const Vt = ws->Vt;
  for(idx_t r=0; r < nrows; ++r) {
    for(idx_t c=0; c < rank; ++c) {
      outmat[c + (r*rank)] = Vt[c + (r*LDVt)];
    }
  }

	timer_stop(&timers[TIMER_SVD]);
}



void alloc_svd_ws(
    svd_ws * const ws,
    idx_t const nmats,
    idx_t const * const nrows,
    idx_t const * const ncolumns)
{
  /* initialize */
  ws->A = NULL;
  ws->U = NULL;
  ws->S = NULL;
  ws->Vt = NULL;
  ws->workspace = NULL;
  ws->iwork = NULL;
  ws->lwork = 0;

  /* allocate SVD workspace */
  idx_t max_A = 0;
  int max_U = 0;
  int max_Vt = 0;
  int max_min = 0;
  int max_lwork = 0;
  for(idx_t m=0; m < nmats; ++m) {
    max_A = SS_MAX(max_A, nrows[m] * ncolumns[m]);

    int const M = (int) ncolumns[m];
    int const N = (int) nrows[m];
    int const mindim = SS_MIN(M, N);
    max_min = SS_MAX(max_min, mindim);

    max_U = SS_MAX(max_U, M * mindim);
    max_Vt = SS_MAX(max_Vt, N * mindim);

    int const lwork = p_optimal_svd_work_size(M, N);
    max_lwork= SS_MAX(lwork, max_lwork);
  }

  /* allocate matrices */
  ws->A = malloc(max_A * sizeof(*(ws->A)));
  ws->U = malloc(max_U * sizeof(*(ws->U)));
  ws->S = malloc(max_min * sizeof(*(ws->S)));
  ws->Vt = malloc(max_Vt * sizeof(*(ws->Vt)));

  /* allocate workspaces */
  ws->lwork = max_lwork;
  ws->workspace = malloc(max_lwork * sizeof(*(ws->workspace)));
  ws->iwork = malloc(8 * max_min * sizeof(*(ws->iwork)));

  ws->P = NULL;
  ws->Q = NULL;
  ws->alphas = NULL;
  ws->betas = NULL;
}


void free_svd_ws(
    svd_ws * const ws)
{
  free(ws->A);
  free(ws->S);
  free(ws->U);
  free(ws->Vt);
  free(ws->workspace);
  free(ws->iwork);
}

