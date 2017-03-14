

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "util.h"
#include "timer.h"
#include "splatt_lapack.h"
#include <math.h>



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Form the Gram matrix from A^T * A.
*
* @param[out] neq_matrix The matrix to fill.
* @param aTa The individual Gram matrices.
* @param mode Which mode we are computing for.
* @param nmodes How many total modes.
* @param reg Regularization parameter (to add to the diagonal).
*/
static void p_form_gram(
    matrix_t * neq_matrix,
    matrix_t * * aTa,
    idx_t const mode,
    idx_t const nmodes,
    val_t const reg)
{
  /* nfactors */
  splatt_blas_int N = aTa[0]->J;

  /* form upper-triangual normal equations */
  val_t * const restrict neqs = neq_matrix->vals;
  #pragma omp parallel
  {
    /* first initialize with 1s */
    #pragma omp for schedule(static, 1)
    for(splatt_blas_int i=0; i < N; ++i) {
      neqs[i+(i*N)] = 1. + reg;
      for(splatt_blas_int j=0; j < N; ++j) {
        neqs[j+(i*N)] = 1.;
      }
    }

    /* now Hadamard product all (A^T * A) matrices */
    for(idx_t m=0; m < nmodes; ++m) {
      if(m == mode) {
        continue;
      }

      val_t const * const restrict mat = aTa[m]->vals;
      #pragma omp for schedule(static, 1)
      for(splatt_blas_int i=0; i < N; ++i) {
        /* 
         * `mat` is symmetric but stored upper right triangular, so be careful
         * to only access that.
         */

        /* copy upper triangle */
        for(splatt_blas_int j=i; j < N; ++j) {
          neqs[j+(i*N)] *= mat[j+(i*N)];
        }
      }
    } /* foreach mode */

    #pragma omp barrier

    /* now copy lower triangular */
    #pragma omp for schedule(static, 1)
    for(splatt_blas_int i=0; i < N; ++i) {
      for(splatt_blas_int j=0; j < i; ++j) {
        neqs[j+(i*N)] = neqs[i+(j*N)];
      }
    }
  } /* omp parallel */
}



static void p_mat_2norm(
  matrix_t * const A,
  val_t * const restrict lambda,
  rank_info * const rinfo,
  thd_info * const thds)
{
  idx_t const I = A->I;
  idx_t const J = A->J;
  val_t * const restrict vals = A->vals;

  #pragma omp parallel
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * const mylambda = (val_t *) thds[tid].scratch[0];
    for(idx_t j=0; j < J; ++j) {
      mylambda[j] = 0;
    }

    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        mylambda[j] += vals[j + (i*J)] * vals[j + (i*J)];
      }
    }

    /* do reduction on partial sums */
    thd_reduce(thds, 0, J, REDUCE_SUM);

    #pragma omp master
    {
#ifdef SPLATT_USE_MPI
      /* now do an MPI reduction to get the global lambda */
      timer_start(&timers[TIMER_MPI_NORM]);
      timer_start(&timers[TIMER_MPI_COMM]);
      MPI_Allreduce(mylambda, lambda, J, SPLATT_MPI_VAL, MPI_SUM, rinfo->comm_3d);
      timer_stop(&timers[TIMER_MPI_COMM]);
      timer_stop(&timers[TIMER_MPI_NORM]);
#else
      memcpy(lambda, mylambda, J * sizeof(*lambda));
#endif
    }

    #pragma omp barrier

    #pragma omp for schedule(static)
    for(idx_t j=0; j < J; ++j) {
      lambda[j] = sqrt(lambda[j]);
    }

    /* do the normalization */
    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        vals[j+(i*J)] /= lambda[j];
      }
    }
  } /* end omp parallel */
}


static void p_mat_maxnorm(
  matrix_t * const A,
  val_t * const restrict lambda,
  rank_info * const rinfo,
  thd_info * const thds)
{
  idx_t const I = A->I;
  idx_t const J = A->J;
  val_t * const restrict vals = A->vals;

  #pragma omp parallel
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * const mylambda = (val_t *) thds[tid].scratch[0];
    for(idx_t j=0; j < J; ++j) {
      mylambda[j] = 0;
    }

    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        mylambda[j] = SS_MAX(mylambda[j], vals[j+(i*J)]);
      }
    }

    /* do reduction on partial maxes */
    thd_reduce(thds, 0, J, REDUCE_MAX);

    #pragma omp master
    {
#ifdef SPLATT_USE_MPI
      /* now do an MPI reduction to get the global lambda */
      timer_start(&timers[TIMER_MPI_NORM]);
      timer_start(&timers[TIMER_MPI_COMM]);
      MPI_Allreduce(mylambda, lambda, J, SPLATT_MPI_VAL, MPI_MAX, rinfo->comm_3d);
      timer_stop(&timers[TIMER_MPI_COMM]);
      timer_stop(&timers[TIMER_MPI_NORM]);
#else
      memcpy(lambda, mylambda, J * sizeof(val_t));
#endif

    }

    #pragma omp barrier

    #pragma omp for schedule(static)
    for(idx_t j=0; j < J; ++j) {
      lambda[j] = SS_MAX(lambda[j], 1.);
    }

    /* do the normalization */
    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        vals[j+(i*J)] /= lambda[j];
      }
    }
  } /* end omp parallel */
}


/**
* @brief Solve the system LX = B.
*
* @param L The lower triangular matrix of coefficients.
* @param B The right-hand side which is overwritten with X.
*/
static void p_mat_forwardsolve(
  matrix_t const * const L,
  matrix_t * const B)
{
  /* check dimensions */
  idx_t const N = L->I;

  val_t const * const restrict lv = L->vals;
  val_t * const restrict bv = B->vals;

  /* first row of X is easy */
  for(idx_t j=0; j < N; ++j) {
    bv[j] /= lv[0];
  }

  /* now do forward substitution */
  for(idx_t i=1; i < N; ++i) {
    /* X(i,f) = B(i,f) - \sum_{j=0}^{i-1} L(i,j)X(i,j) */
    for(idx_t j=0; j < i; ++j) {
      for(idx_t f=0; f < N; ++f) {
        bv[f+(i*N)] -= lv[j+(i*N)] * bv[f+(j*N)];
      }
    }
    for(idx_t f=0; f < N; ++f) {
      bv[f+(i*N)] /= lv[i+(i*N)];
    }
  }
}


/**
* @brief Solve the system UX = B.
*
* @param U The upper triangular matrix of coefficients.
* @param B The right-hand side which is overwritten with X.
*/
static void p_mat_backwardsolve(
  matrix_t const * const U,
  matrix_t * const B)
{
  /* check dimensions */
  idx_t const N = U->I;

  val_t const * const restrict rv = U->vals;
  val_t * const restrict bv = B->vals;

  /* last row of X is easy */
  for(idx_t f=0; f < N; ++f) {
    idx_t const i = N-1;
    bv[f+(i*N)] /= rv[i+(i*N)];
  }

  /* now do backward substitution */
  for(idx_t row=2; row <= N; ++row) {
    /* operate with (N - row) to make unsigned comparisons easy */
    idx_t const i = N - row;

    /* X(i,f) = B(i,f) - \sum_{j=0}^{i-1} R(i,j)X(i,j) */
    for(idx_t j=i+1; j < N; ++j) {
      for(idx_t f=0; f < N; ++f) {
        bv[f+(i*N)] -= rv[j+(i*N)] * bv[f+(j*N)];
      }
    }
    for(idx_t f=0; f < N; ++f) {
      bv[f+(i*N)] /= rv[i+(i*N)];
    }
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mat_syminv(
  matrix_t * const A)
{
  /* check dimensions */
  assert(A->I == A->J);

  idx_t const N = A->I;

  matrix_t * L = mat_alloc(N, N);

  /* do a Cholesky factorization on A */
  //mat_cholesky(A, L);

  /* setup identity matrix */
  memset(A->vals, 0, N*N*sizeof(val_t));
  for(idx_t n=0; n < N; ++n) {
    A->vals[n+(n*N)] = 1.;
  }

  /* Solve L*Y = I */
  p_mat_forwardsolve(L, A);

  /* transpose L */
  for(idx_t i=0; i < N; ++i) {
    for(idx_t j=i+1; j < N; ++j) {
      L->vals[j+(i*N)] = L->vals[i+(j*N)];
      L->vals[i+(j*N)] = 0.;
    }
  }

  /* Solve U*A = Y */
  p_mat_backwardsolve(L, A);

  mat_free(L);
}


void mat_cholesky(
  matrix_t const * const A)
{
  timer_start(&timers[TIMER_CHOLESKY]);
  /* check dimensions */
  assert(A->I == A->J);

  /* Cholesky factorization */
  int N = (int) A->I;
  val_t * const restrict neqs = A->vals;
  char uplo = 'L';
  int order = N;
  int lda = N;
  int info;
  LAPACK_DPOTRF(&uplo, &order, neqs, &lda, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRF returned %d\n", info);
  }

  timer_stop(&timers[TIMER_CHOLESKY]);
}


void mat_solve_cholesky(
    matrix_t * const cholesky,
    matrix_t * const rhs)
{
  /* Chunked AO-ADMM will call this from a parallel region. */
  if(!splatt_omp_in_parallel()) {
    timer_start(&timers[TIMER_BACKSOLVE]);
  }
  int N = (int) cholesky->I;

  /* Solve against rhs */
  char tri = 'L';
  int lda = N;
  int info;
  int nrhs = (int) rhs->I;
  int ldb = N;
  LAPACK_DPOTRS(&tri, &N, &nrhs, cholesky->vals, &lda, rhs->vals, &ldb, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRS returned %d\n", info);
  }

  if(!splatt_omp_in_parallel()) {
    timer_stop(&timers[TIMER_BACKSOLVE]);
  }
}


val_t mat_trace(
    matrix_t const * const A)
{
  assert(A->I == A->J);

  idx_t const N = A->I;
  val_t const * const restrict vals = A->vals;

  val_t trace = 0.;
  for(idx_t i=0; i < N; ++i) {
    trace += vals[i + (i*N)];
  }

  return trace;
}


void mat_aTa_hada(
  matrix_t ** mats,
  idx_t const start,
  idx_t const nmults,
  idx_t const nmats,
  matrix_t * const buf,
  matrix_t * const ret)
{
  idx_t const F = mats[0]->J;

  /* check matrix dimensions */
  assert(ret->I == ret->J);
  assert(ret->I == F);
  assert(buf->I == F);
  assert(buf->J == F);
  assert(ret->vals != NULL);
  assert(mats[0]->rowmajor);
  assert(ret->rowmajor);

  val_t       * const restrict rv   = ret->vals;
  val_t       * const restrict bufv = buf->vals;
  for(idx_t i=0; i < F; ++i) {
    for(idx_t j=i; j < F; ++j) {
      rv[j+(i*F)] = 1.;
    }
  }

  for(idx_t mode=0; mode < nmults; ++mode) {
    idx_t const m = (start+mode) % nmats;
    idx_t const I  = mats[m]->I;
    val_t const * const Av = mats[m]->vals;
    memset(bufv, 0, F * F * sizeof(val_t));

    /* compute upper triangular matrix */
    for(idx_t i=0; i < I; ++i) {
      for(idx_t mi=0; mi < F; ++mi) {
        for(idx_t mj=mi; mj < F; ++mj) {
          bufv[mj + (mi*F)] += Av[mi + (i*F)] * Av[mj + (i*F)];
        }
      }
    }

    /* hadamard product */
    for(idx_t mi=0; mi < F; ++mi) {
      for(idx_t mj=mi; mj < F; ++mj) {
        rv[mj + (mi*F)] *= bufv[mj + (mi*F)];
      }
    }
  }

  /* copy to lower triangular matrix */
  for(idx_t i=1; i < F; ++i) {
    for(idx_t j=0; j < i; ++j) {
      rv[j + (i*F)] = rv[i + (j*F)];
    }
  }
}


void mat_aTa(
  matrix_t const * const A,
  matrix_t * const ret,
  rank_info * const rinfo)
{
  timer_start(&timers[TIMER_ATA]);
  /* check matrix dimensions */
  assert(ret->I == ret->J);
  assert(ret->I == A->J);
  assert(ret->vals != NULL);
  assert(A->rowmajor);
  assert(ret->rowmajor);

  idx_t const I = A->I;
  idx_t const F = A->J;
  val_t const * const restrict Av = A->vals;

  char uplo = 'L';
  char trans = 'N'; /* actually do A * A' due to row-major ordering */
  splatt_blas_int N = (splatt_blas_int) F;
  splatt_blas_int K = (splatt_blas_int) I;
  splatt_blas_int lda = N;
  splatt_blas_int ldc = N;
  val_t alpha = 1.;
  val_t beta = 0.;

  SPLATT_BLAS(syrk)(&uplo, &trans, &N, &K, &alpha, A->vals, &lda, &beta, ret->vals,
      &ldc);

#ifdef SPLATT_USE_MPI
  timer_start(&timers[TIMER_MPI_ATA]);
  timer_start(&timers[TIMER_MPI_COMM]);
  MPI_Allreduce(MPI_IN_PLACE, ret->vals, F * F, SPLATT_MPI_VAL, MPI_SUM,
      rinfo->comm_3d);
  timer_stop(&timers[TIMER_MPI_COMM]);
  timer_stop(&timers[TIMER_MPI_ATA]);
#endif

  timer_stop(&timers[TIMER_ATA]);
}

void mat_matmul(
  matrix_t const * const A,
  matrix_t const * const B,
  matrix_t  * const C)
{
  timer_start(&timers[TIMER_MATMUL]);
  /* check dimensions */
  assert(A->J == B->I);
  assert(C->I * C->J <= A->I * B->J);

  /* set dimensions */
  C->I = A->I;
  C->J = B->J;

  val_t const * const restrict av = A->vals;
  val_t const * const restrict bv = B->vals;
  val_t       * const restrict cv = C->vals;

  idx_t const M  = A->I;
  idx_t const N  = B->J;
  idx_t const Na = A->J;

  /* tiled matrix multiplication */
  idx_t const TILE = 16;
  #pragma omp parallel for schedule(static)
  for(idx_t i=0; i < M; ++i) {
    for(idx_t jt=0; jt < N; jt += TILE) {
      for(idx_t kt=0; kt < Na; kt += TILE) {
        idx_t const JSTOP = SS_MIN(jt+TILE, N);
        for(idx_t j=jt; j < JSTOP; ++j) {
          val_t accum = 0;
          idx_t const KSTOP = SS_MIN(kt+TILE, Na);
          for(idx_t k=kt; k < KSTOP; ++k) {
            accum += av[k + (i*Na)] * bv[j + (k*N)];
          }
          cv[j + (i*N)] += accum;
        }
      }
    }
  }

  timer_stop(&timers[TIMER_MATMUL]);
}

void mat_normalize(
  matrix_t * const A,
  val_t * const restrict lambda,
  splatt_mat_norm const which,
  rank_info * const rinfo,
  thd_info * const thds)
{
  timer_start(&timers[TIMER_MATNORM]);

  switch(which) {
  case MAT_NORM_2:
    p_mat_2norm(A, lambda, rinfo, thds);
    break;
  case MAT_NORM_MAX:
    p_mat_maxnorm(A, lambda, rinfo, thds);
    break;
  default:
    fprintf(stderr, "SPLATT: mat_normalize supports 2 and MAX only.\n");
    abort();
  }
  timer_stop(&timers[TIMER_MATNORM]);
}


void mat_form_gram(
    matrix_t * * aTa,
    matrix_t * out_mat,
    idx_t nmodes,
    idx_t mode)
{
  idx_t const N = aTa[mode]->J;
  val_t * const restrict gram = out_mat->vals;

  #pragma omp parallel
  {
    /* first initialize */
    #pragma omp for schedule(static, 1)
    for(idx_t i=0; i < N; ++i) {
      for(idx_t j=i; j < N; ++j) {
        gram[j+(i*N)] = 1.;
      }
    }

    for(idx_t m=0; m < nmodes; ++m) {
      if(m == mode) {
        continue;
      }

      val_t const * const restrict mat = aTa[m]->vals;
      #pragma omp for schedule(static, 1) nowait
      for(idx_t i=0; i < N; ++i) {
        for(idx_t j=i; j < N; ++j) {
          gram[j+(i*N)] *= mat[j+(i*N)];
        }
      }
    }
  } /* omp parallel */
}


void mat_solve_normals(
  idx_t const mode,
  idx_t const nmodes,
	matrix_t * * aTa,
  matrix_t * rhs,
  val_t const reg)
{
  timer_start(&timers[TIMER_INV]);

  /* nfactors */
  splatt_blas_int N = aTa[0]->J;

  p_form_gram(aTa[MAX_NMODES], aTa, mode, nmodes, reg);

  splatt_blas_int info;
  char uplo = 'L';
  splatt_blas_int lda = N;
  splatt_blas_int ldb = N;
  splatt_blas_int order = N;
  splatt_blas_int nrhs = (splatt_blas_int) rhs->I;

  val_t * const neqs = aTa[MAX_NMODES]->vals;

  /* Cholesky factorization */
  bool is_spd = true;
  SPLATT_BLAS(potrf)(&uplo, &order, neqs, &lda, &info);
  if(info) {
    fprintf(stderr, "SPLATT: Gram matrix is not SPD. Trying `GELSS`.\n");
    is_spd = false;
  }

  /* Continue with Cholesky */
  if(is_spd) {
    /* Solve against rhs */
    SPLATT_BLAS(potrs)(&uplo, &order, &nrhs, neqs, &lda, rhs->vals, &ldb, &info);
    if(info) {
      fprintf(stderr, "SPLATT: DPOTRS returned %d\n", info);
    }
  } else {
    /* restore gram matrix */
    p_form_gram(aTa[MAX_NMODES], aTa, mode, nmodes, reg);

    splatt_blas_int effective_rank;
    val_t * conditions = splatt_malloc(N * sizeof(*conditions));

    /* query worksize */
    splatt_blas_int lwork = -1;

    val_t rcond = -1.0f;

    val_t work_query;
    SPLATT_BLAS(gelss)(&N, &N, &nrhs,
        neqs, &lda,
        rhs->vals, &ldb,
        conditions, &rcond, &effective_rank,
        &work_query, &lwork, &info);
    lwork = (splatt_blas_int) work_query;

    /* setup workspace */
    val_t * work = splatt_malloc(lwork * sizeof(*work));

    /* Use an SVD solver */
    SPLATT_BLAS(gelss)(&N, &N, &nrhs,
        neqs, &lda,
        rhs->vals, &ldb,
        conditions, &rcond, &effective_rank,
        work, &lwork, &info);
    if(info) {
      printf("SPLATT: DGELSS returned %d\n", info);
    }
    printf("SPLATT:   DGELSS effective rank: %d\n", effective_rank);

    splatt_free(conditions);
    splatt_free(work);
  }

  timer_stop(&timers[TIMER_INV]);
}


void mat_add_diag(
    matrix_t * const A,
    val_t const scalar)
{
  idx_t const rank = A->J;
  val_t * const restrict vals = A->vals;

  for(idx_t i=0; i < rank; ++i) {
    vals[i + (i*rank)] += scalar;
  }
}



void calc_gram_inv(
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



matrix_t * mat_alloc(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = (matrix_t *) splatt_malloc(sizeof(matrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->vals = (val_t *) splatt_malloc(nrows * ncols * sizeof(val_t));
  mat->rowmajor = 1;
  return mat;
}

matrix_t * mat_rand(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = mat_alloc(nrows, ncols);
  val_t * const vals = mat->vals;

  fill_rand(vals, nrows * ncols);

  return mat;
}


matrix_t * mat_zero(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = mat_alloc(nrows, ncols);

  /* Initialize in parallel in case system is NUMA. This may bring a small
   * improvement. */
  #pragma omp parallel for schedule(static)
  for(idx_t i=0; i < nrows; ++i) {
    for(idx_t j=0; j < ncols; ++j) {
      mat->vals[j + (i*ncols)] = 0.;
    }
  }

  return mat;
}


matrix_t * mat_mkptr(
    val_t * const data,
    idx_t rows,
    idx_t cols,
    int rowmajor)
{
  matrix_t * mat = splatt_malloc(sizeof(*mat));

  mat_fillptr(mat, data, rows, cols, rowmajor);

  return mat;
}


void mat_fillptr(
    matrix_t * ptr,
    val_t * const data,
    idx_t rows,
    idx_t cols,
    int rowmajor)
{
  ptr->I = rows;
  ptr->J = cols;
  ptr->rowmajor = rowmajor;
  ptr->vals = data;
}


void mat_free(
  matrix_t * mat)
{
  if(mat == NULL) {
    return;
  }
  splatt_free(mat->vals);
  splatt_free(mat);
}


matrix_t * mat_mkrow(
  matrix_t const * const mat)
{
  assert(mat->rowmajor == 0);

  idx_t const I = mat->I;
  idx_t const J = mat->J;

  matrix_t * row = mat_alloc(I, J);
  val_t       * const restrict rowv = row->vals;
  val_t const * const restrict colv = mat->vals;

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      rowv[j + (i*J)] = colv[i + (j*I)];
    }
  }

  return row;
}

matrix_t * mat_mkcol(
  matrix_t const * const mat)
{
  assert(mat->rowmajor == 1);
  idx_t const I = mat->I;
  idx_t const J = mat->J;

  matrix_t * col = mat_alloc(I, J);
  val_t       * const restrict colv = col->vals;
  val_t const * const restrict rowv = mat->vals;

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      colv[i + (j*I)] = rowv[j + (i*J)];
    }
  }

  col->rowmajor = 0;

  return col;
}


spmatrix_t * spmat_alloc(
  idx_t const nrows,
  idx_t const ncols,
  idx_t const nnz)
{
  spmatrix_t * mat = (spmatrix_t*) splatt_malloc(sizeof(spmatrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->nnz = nnz;
  mat->rowptr = (idx_t*) splatt_malloc((nrows+1) * sizeof(idx_t));
  mat->colind = (idx_t*) splatt_malloc(nnz * sizeof(idx_t));
  mat->vals   = (val_t*) splatt_malloc(nnz * sizeof(val_t));
  return mat;
}

void spmat_free(
  spmatrix_t * mat)
{
  free(mat->rowptr);
  free(mat->colind);
  free(mat->vals);
  free(mat);
}

