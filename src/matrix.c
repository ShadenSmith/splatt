

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "util.h"

#include <math.h>



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mat_matmul(
  matrix_t const * const A,
  matrix_t const * const B,
  matrix_t  * const C)
{
  /* check dimensions */
  assert(A->J == B->I);
  assert(C->I == A->I);
  assert(C->J == B->J);

  idx_t const I = A->I;
  idx_t const J = B->J;
  idx_t const aj = A->J;

  val_t * const restrict cv = C->vals;
  memset(cv, 0, I * J * sizeof(val_t));

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      for(idx_t ii=0; ii < aj; ++ii) {
      }
    }
  }

}


void mat_syminv(
  matrix_t * const A,
  matrix_t * const Abuf)
{

}



void mat_aTa_hada(
  matrix_t ** mats,
  idx_t const start,
  idx_t const end,
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

  for(idx_t m=start; m != end; m = (m+1) % nmats) {
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
  matrix_t * const ret)
{
  /* check matrix dimensions */
  assert(ret->I == ret->J);
  assert(ret->I == A->J);
  assert(ret->vals != NULL);
  assert(A->rowmajor);
  assert(ret->rowmajor);

  idx_t const I = A->I;
  idx_t const F = A->J;
  val_t const * const restrict Av = A->vals;
  val_t       * const restrict rv = ret->vals;

  /* compute upper triangular portion */
  memset(rv, 0, F * F * sizeof(val_t));
  for(idx_t i=0; i < I; ++i) {
    for(idx_t mi=0; mi < F; ++mi) {
      for(idx_t mj=mi; mj < F; ++mj) {
        rv[mj + (mi*F)] += Av[mi + (i*F)] * Av[mj + (i*F)];
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

void mat_normalize(
  matrix_t * const A,
  val_t * const lambda,
  splatt_mat_norm const which)
{
  idx_t const I = A->I;
  idx_t const J = A->J;
  val_t * const restrict vals = A->vals;

  assert(vals != NULL);
  assert(lambda != NULL);

  for(idx_t j=0; j < J; ++j) {
    lambda[j] = 0;
  }

  /* get column norms */
  switch(which) {
  case MAT_NORM_2:
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        lambda[j] += vals[j + (i*J)] * vals[j + (i*J)];
      }
    }
    for(idx_t j=0; j < J; ++j) {
      lambda[j] = sqrt(lambda[j]);
    }
    break;

  case MAT_NORM_MAX:
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        lambda[j] = (vals[j+(i*J)] > lambda[j]) ?  vals[j+(i*J)] : lambda[j];
      }
    }
    for(idx_t j=0; j < J; ++j) {
      lambda[j] = (lambda[j] > 1.) ? lambda[j] : 1.;
    }
    break;
  }

  /* do the normalization */
  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      vals[j+(i*J)] /= lambda[j];
    }
  }
}



matrix_t * mat_alloc(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = (matrix_t *) malloc(sizeof(matrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->vals = (val_t *) malloc(nrows * ncols * sizeof(val_t));
  mat->rowmajor = 1;
  return mat;
}

matrix_t * mat_rand(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = mat_alloc(nrows, ncols);
  val_t * const vals = mat->vals;

  for(idx_t i=0; i < nrows; ++i) {
    for(idx_t j=0; j < ncols; ++j) {
      vals[j + (i*ncols)] = rand_val();
    }
  }

  return mat;
}

void mat_free(
  matrix_t * mat)
{
  free(mat->vals);
  free(mat);
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
  spmatrix_t * mat = (spmatrix_t*) malloc(sizeof(spmatrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->nnz = nnz;
  mat->rowptr = (idx_t*) malloc((nrows+1) * sizeof(idx_t));
  mat->colind = (idx_t*) malloc(nnz * sizeof(idx_t));
  mat->vals   = (val_t*) malloc(nnz * sizeof(val_t));
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

