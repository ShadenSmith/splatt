
#include "../src/matrix.h"
#include "ctest/ctest.h"
#include "splatt_test.h"
#include <math.h>
#include <omp.h>

#define NMATS 4

CTEST_DATA(matrix)
{
  idx_t nthreads;
  matrix_t * mats[NMATS];
};

CTEST_SETUP(matrix)
{
  data->nthreads = 7;
  omp_set_num_threads(data->nthreads);

  data->mats[0] = mat_rand(100, 3);
  data->mats[1] = mat_rand(3, 100);
  data->mats[2] = mat_rand(100, 100);
  data->mats[3] = mat_rand(100, 1);
}

CTEST_TEARDOWN(matrix)
{
  for(idx_t m=0; m < NMATS; ++m) {
    mat_free(data->mats[m]);
  }
}

CTEST2(matrix, transpose)
{
  for(idx_t m=0; m < NMATS; ++m) {
    matrix_t const * const A = data->mats[m];
    matrix_t * B = mat_alloc(A->I, A->J);

    mat_transpose(A, B);


    idx_t const I = A->I;
    idx_t const J = A->J;

    ASSERT_EQUAL(I, B->J);
    ASSERT_EQUAL(J, B->I);

    val_t const * const av = A->vals;
    val_t const * const bv = B->vals;

    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        ASSERT_DBL_NEAR_TOL(av[j + (i * A->J)], bv[i + (j * B->J)], 0);
      }
    }

    mat_free(B);
  }
}


CTEST2(matrix, matmul)
{
  for(idx_t m=0; m < NMATS; ++m) {
    matrix_t const * const A = data->mats[m];

    idx_t const K = A->J;
    matrix_t * B = mat_rand(K, 19);

    /* perform matrix multiplication */
    matrix_t * C = mat_alloc(B->J, A->I);
    memset(C->vals, 0, C->I * C->J * sizeof(val_t));
    mat_matmul(A, B, C);

    idx_t const I = A->I;
    idx_t const J = B->J;

    /* check dimensions */
    ASSERT_EQUAL(I, C->I);
    ASSERT_EQUAL(J, C->J);


    matrix_t * gold = mat_alloc(A->I, B->J);
    memset(gold->vals, 0, gold->I * gold->J * sizeof(val_t));

    val_t const * const av = A->vals;
    val_t const * const bv = B->vals;
    val_t * gv = gold->vals;

    /* compute gold */
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        for(idx_t k=0; k < K; ++k) {
          gv[j + (i*J)] += av[k+(i*K)] * bv[j + (k*J)];
        }
      }
    }

    /* compare */
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        ASSERT_DBL_NEAR_TOL(gv[j+(i*J)], C->vals[j+(i*J)], 1e-12);
      }
    }

    mat_free(B);
    mat_free(C);
    mat_free(gold);
  }
}


CTEST2(matrix, ata)
{
  for(idx_t m=0; m < NMATS; ++m) {
    matrix_t const * const A = data->mats[m];

    idx_t const F = A->J;
    matrix_t * B = mat_alloc(F, F);

    thd_info * thds = thd_init(data->nthreads, 1, F * F * sizeof(val_t));

    mat_aTa(A, B, NULL, thds, data->nthreads);

    matrix_t * At = mat_alloc(A->J, A->I);
    mat_transpose(A, At);

    /* actually compute A^T * A */
    matrix_t * gold = mat_alloc(F, F);
    memset(gold->vals, 0, F * F * sizeof(val_t));
    mat_matmul(At, A, gold);

    /* compare upper triangular */
    for(idx_t i=0; i < F; ++i) {
      for(idx_t j=i; j < F; ++j) {
        ASSERT_DBL_NEAR_TOL(gold->vals[j+(i*F)], B->vals[j+(i*F)], 1e-12);
      }
    }

    thd_free(thds, data->nthreads);
    mat_free(B);
    mat_free(gold);
  }
}


CTEST2(matrix, mat_col_orth)
{
  /* normalize first column */
  matrix_t * A = mat_mkcol(data->mats[2]);

  val_t norm = 0;
  for(idx_t i=0; i < A->I; ++i) {
    norm += A->vals[i] * A->vals[i];
  }
  norm = sqrt(norm);
  for(idx_t i=0; i < A->I; ++i) {
    A->vals[i] /= norm;
  }

  /* now orthogonalize the second column */
  mat_col_orth(A, 1);

  /* check the inner product */
  val_t inner = 0;
  for(idx_t i=0; i < A->I; ++i) {
    inner += A->vals[i] * A->vals[i + A->I];
  }
  ASSERT_DBL_NEAR_TOL(0., inner, 1e-12);

  /* now do next column */
  mat_col_orth(A, 2);

  inner = 0;
  val_t inner2 = 0;
  for(idx_t i=0; i < A->I; ++i) {
    inner  += A->vals[i] * A->vals[i + (1 * A->I)];
    inner2 += A->vals[i] * A->vals[i + (2 * A->I)];
  }
  ASSERT_DBL_NEAR_TOL(0., inner, 1e-12);
  ASSERT_DBL_NEAR_TOL(0., inner2, 1e-12);

  mat_free(A);
}


CTEST2(matrix, mat_col_orth_big)
{
  /* normalize first column */
  matrix_t * A = mat_rand(500, 101);
  A->rowmajor = 0;

  for(idx_t j=0; j < A->J; ++j) {
    /* now orthogonalize the second column */
    mat_col_orth(A, j);

    vec_normalize(A->vals + (j * A->I), A->I);

    /* check that the inner products are 0 */
    for(idx_t k=0; k <= j; ++k) {
      val_t const * const restrict u = A->vals + (k * A->I);
      val_t const * const restrict v = A->vals + (j * A->I);
      val_t inner = 0;
      for(idx_t i=0; i < A->I; ++i) {
        inner += u[i] * v[i];
      }
      if(j == k) {
        ASSERT_DBL_NEAR_TOL(1., inner, 1e-12);
      } else {
        ASSERT_DBL_NEAR_TOL(0., inner, 1e-12);
      }
    }
  }

  mat_free(A);
}


CTEST(matrix, vec_normalize)
{
  idx_t N = 100;
  val_t * vec = splatt_malloc(N * sizeof(*vec));
  val_t x = 0.;
  for(idx_t i=0; i < N; ++i) {
    vec[i] = i;
    x += i * i;
  }

  val_t norm = vec_normalize(vec, N);

  ASSERT_DBL_NEAR_TOL(sqrt(x), norm, 0);

  val_t observed = 0.;
  for(idx_t i=0; i < N; ++i) {
    observed += vec[i] * vec[i];
  }

  ASSERT_DBL_NEAR_TOL(1., sqrt(observed), 1e-16);
}

