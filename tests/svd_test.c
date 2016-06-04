
#include "../src/svd.h"
#include "ctest/ctest.h"

#include <math.h>

#include "splatt_test.h"


/* 2 left singular vectors of [1 2 3; 4 5 6; 7 8 9; 10 11 12] */
val_t goldU[] = {
  -0.140877,   0.824714,
  -0.343946,   0.426264,
  -0.547016,   0.027814,
  -0.750086,  -0.370637
};

CTEST_DATA(svd)
{
  idx_t nrows;
  idx_t ncols;
  idx_t rank;
  val_t * A;
  val_t * buf;
  val_t * left;

  matrix_t matA;
};


CTEST_SETUP(svd)
{
  data->nrows = 4;
  data->ncols = 3;
  data->A = malloc(data->nrows * data->ncols * sizeof(val_t));
  data->left = malloc(data->nrows * data->ncols * sizeof(val_t));
  for(idx_t i=0; i < data->nrows * data->ncols; ++i) {
    data->A[i] = i+1;
  }

  data->matA.I = data->nrows;
  data->matA.J = data->ncols;
  data->matA.vals = data->A;
  data->matA.rowmajor = 1;
}


CTEST_TEARDOWN(svd)
{
  free(data->A);
  free(data->left);
}

CTEST2(svd, svd)
{
  svd_ws ws;
  alloc_svd_ws(&ws, 1, &(data->nrows), &(data->ncols));

  for(idx_t r=1; r <= 2; ++r) {
    left_singulars(data->A, data->left, data->nrows, data->ncols, r, &ws);


    for(idx_t i=0; i < data->nrows; ++i) {
      for(idx_t j=0; j < r-1; ++j) {
        ASSERT_DBL_NEAR_TOL(goldU[j+(i*r)], data->left[j+(i*r)], 1e-6);
      }
    }
  }

  free_svd_ws(&ws);
}

#include "../src/io.h"

CTEST2(svd, lanczos_bidiag)
{
  svd_ws ws;
  alloc_svd_ws(&ws, 1, &(data->nrows), &(data->ncols));

  idx_t const rank = 5;
  matrix_t * A = mat_rand(20,rank);

  lanczos_bidiag(A, rank, &ws);

  /* fill in B */
  matrix_t * B = mat_alloc(rank, rank);
  memset(B->vals, 0, rank * rank * sizeof(*B->vals));
  for(idx_t i=0; i < rank; ++i) {
    B->vals[i + (i*rank)] = ws.alphas[i];
    if(i != rank-1) {
      B->vals[(i+1) + (i*rank)] = ws.betas[i];
    }
  }

  matrix_t * P = mat_mkrow(ws.P);

  /* just copy column major version to get Q^T */
  matrix_t * Q = mat_alloc(rank, rank);
  memcpy(Q->vals, ws.Q->vals, rank * rank * sizeof(*Q->vals));


  /* now reconstruct A */
  matrix_t * tmp = mat_alloc(A->I, rank);
  memset(tmp->vals, 0, A->I * rank * sizeof(*tmp->vals));

  /* P * B */
  mat_matmul(P, B, tmp);

  /* A_new = tmp * Q^T */
  matrix_t * A_new = mat_alloc(A->I, A->J);
  memset(A_new->vals, 0, A->I * A->J * sizeof(*A_new->vals));
  mat_matmul(tmp, Q, A_new);

  val_t error = 0.;
  for(idx_t x=0; x < A->I * A->J; ++x) {
    val_t diff = A->vals[x] - A_new->vals[x];
    error += diff * diff;
  }

#if SPLATT_VAL_TYPEWIDTH == 32
  ASSERT_DBL_NEAR_TOL(0., sqrt(error), 1.5e-3);
#else
  ASSERT_DBL_NEAR_TOL(0., sqrt(error), 1e-12);
#endif

  mat_free(P);
  mat_free(B);
  mat_free(Q);
  mat_free(tmp);
  mat_free(A_new);
  free_svd_ws(&ws);
}


/* compares against two-sided Lanczos bidiagonalization */
CTEST2(svd, lanczos_onesided_bidiag)
{
  svd_ws ws, ws2;
  alloc_svd_ws(&ws, 1, &(data->nrows), &(data->ncols));
  alloc_svd_ws(&ws2, 1, &(data->nrows), &(data->ncols));

  srand(1);
  idx_t const rank = 3;
  matrix_t * A = mat_rand(20,rank);

  /* srand to ensure same initialization */
  srand(1);
  lanczos_bidiag(A, rank, &ws);

  srand(1);
  lanczos_onesided_bidiag(A, rank, &ws2);

  /* Compare B's */
  for(idx_t i=0; i < rank; ++i) {
#if SPLATT_VAL_TYPEWIDTH == 32
    ASSERT_DBL_NEAR_TOL(ws.alphas[i], ws2.alphas[i], 1e-6);
#else
    ASSERT_DBL_NEAR_TOL(ws.alphas[i], ws2.alphas[i], 1e-14);
#endif
    if(i != rank-1) {
#if SPLATT_VAL_TYPEWIDTH == 32
      ASSERT_DBL_NEAR_TOL(ws.betas[i], ws2.betas[i], 1e-6);
#else
      ASSERT_DBL_NEAR_TOL(ws.betas[i], ws2.betas[i], 1e-14);
#endif
    }
  }

  /* Compare Q's */
  for(idx_t x=0; x < rank*rank; ++x) {
#if SPLATT_VAL_TYPEWIDTH == 32
    ASSERT_DBL_NEAR_TOL(ws.Q->vals[x], ws2.Q->vals[x], 3e-6);
#else
    ASSERT_DBL_NEAR_TOL(ws.Q->vals[x], ws2.Q->vals[x], 1e-14);
#endif
  }

  mat_free(A);
  free_svd_ws(&ws);
  free_svd_ws(&ws2);
}



CTEST2(svd, fast_svd)
{
  svd_ws ws;
  alloc_svd_ws(&ws, 1, &(data->nrows), &(data->ncols));

  printf("\n");
  mat_write(&data->matA, NULL);

  fast_left_singulars(&data->matA, 2, &ws);

  printf("Left singular vectors:\n");
  mat_write(ws.P, NULL);
  printf("\n");

  printf("\n");
  mat_write(ws.Q, NULL);
  printf("\n");

  //free_svd_ws(&ws);
}
