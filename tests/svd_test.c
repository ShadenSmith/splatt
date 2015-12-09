
#include "../src/svd.h"
#include "ctest/ctest.h"

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
