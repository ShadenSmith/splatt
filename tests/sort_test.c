
#include "../src/sptensor.h"
#include "../src/io.h"
#include "../src/sort.h"
#include "../src/tile.h"
#include "../src/util.h"

#include "ctest/ctest.h"

#include "splatt_test.h"
#include <omp.h>


/**
* @brief Return 1 if inds[a] <= inds[b].
*
* @param inds The index arrays of a tensor.
* @param m The most important mode.
* @param a Nonzero that should be less than b.
* @param b The (should be) greater nonzero.
*
* @return 1 if a < b, otherwise 0.
*/
static int __idx_cmp(
  idx_t ** inds,
  idx_t const mode,
  idx_t const nmodes,
  idx_t const a,
  idx_t const b)
{
  if(inds[mode][a] < inds[mode][b]) {
    return 1;
  }

  if(inds[mode][a] > inds[mode][b]) {
    return 0;
  }

  for(idx_t m=1; m < nmodes; ++m) {
    if(inds[(m+mode)%nmodes][a] < inds[(m+mode)%nmodes][b]) {
      return 1;
    }
  }

  /* last chance, check for equality */
  for(idx_t m=0; m < nmodes; ++m) {
    if(inds[m][a] != inds[m][b]) {
      return 0;
    }
  }

  return 1;
}


CTEST_DATA(sort_tensor)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
};


CTEST_SETUP(sort_tensor)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}

CTEST_TEARDOWN(sort_tensor)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}


CTEST2(sort_tensor, full_sort)
{
  omp_set_num_threads(2);
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];

    for(idx_t m=tt->nmodes; m-- != 0; ) {
      tt_sort(tt, m, NULL);

      for(idx_t x=0; x < tt->nnz-1; ++x) {
        ASSERT_EQUAL(1, __idx_cmp(tt->ind, m, tt->nmodes, x, x+1));
      }
    }

    return;
  }
}

CTEST2(sort_tensor, par_sort)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * gold = data->tensors[i];

    /* make a copy */
    sptensor_t * test = tt_alloc(gold->nnz, gold->nmodes);
    memcpy(test->dims, gold->dims, gold->nmodes * sizeof(*(test->dims)));
    for(idx_t m=0; m < gold->nmodes; ++m) {
      memcpy(test->ind[m], gold->ind[m], gold->nnz * sizeof(**(test->ind)));
    }
    memcpy(test->vals, gold->vals, gold->nnz * sizeof(*(test->vals)));

    for(idx_t m=gold->nmodes; m-- != 0; ) {
      omp_set_num_threads(1);
      tt_sort(gold, m, NULL);

      omp_set_num_threads(24);
      tt_sort(test, m, NULL);

      /* better be exactly the same! */
      for(idx_t m=0; m < test->nmodes; ++m) {
        idx_t const * const gold_ind = gold->ind[m];
        idx_t const * const test_ind = test->ind[m];

        for(idx_t n=0; n < test->nnz; ++n) {
          ASSERT_EQUAL(gold_ind[n], test_ind[n]);
        }
      }
      val_t const * const gold_vals = gold->vals;
      val_t const * const test_vals = test->vals;
      for(idx_t n=0; n < test->nnz; ++n) {
        ASSERT_DBL_NEAR_TOL(gold_vals[n], test_vals[n], 0.);
      }
    }

    tt_free(test);
  }
}


CTEST2(sort_tensor, tiled_sort)
{
  idx_t tdims[MAX_NMODES];

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];

    idx_t ntiles = 1;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      tdims[m] = 5;
      ntiles *= tdims[m];
    }
    idx_t * nnzptr = tt_densetile(tt, tdims);

    for(idx_t t=0; t < ntiles; ++t) {
      for(idx_t m=tt->nmodes; m-- != 0; ) {
        tt_sort_range(tt, m, NULL, nnzptr[t], nnzptr[t+1]);

        for(idx_t x=nnzptr[t]; x < nnzptr[t]; ++x) {
          ASSERT_EQUAL(1, __idx_cmp(tt->ind, m, tt->nmodes, x, x+1));
        }
      }
    }
    free(nnzptr);
  }
}


CTEST_DATA(sort_idx)
{
  idx_t N;
  idx_t * rand_idx;
  idx_t * fororder;
  idx_t * revorder;
};


CTEST_SETUP(sort_idx)
{
  data->N = 100000;
  data->rand_idx = splatt_malloc(data->N * sizeof(*data->rand_idx));
  data->fororder = splatt_malloc(data->N * sizeof(*data->fororder));
  data->revorder = splatt_malloc(data->N * sizeof(*data->revorder));

  for(idx_t x=0; x < data->N; ++x) {
    data->rand_idx[x] = rand_idx();
    data->fororder[x] = x;
    data->revorder[x] = data->N - x;
  }
}

CTEST_TEARDOWN(sort_idx)
{
  splatt_free(data->rand_idx);
  splatt_free(data->fororder);
  splatt_free(data->revorder);
}


CTEST2(sort_idx, qsort)
{
  quicksort(data->rand_idx, data->N);
  for(idx_t x=0; x < data->N - 1; ++x) {
    if(data->rand_idx[x] > data->rand_idx[x+1]) {
      ASSERT_FAIL();
    }
  }

  quicksort(data->fororder, data->N);
  for(idx_t x=0; x < data->N - 1; ++x) {
    if(data->fororder[x] > data->fororder[x+1]) {
      ASSERT_FAIL();
    }
  }

  quicksort(data->revorder, data->N);
  for(idx_t x=0; x < data->N - 1; ++x) {
    if(data->revorder[x] > data->revorder[x+1]) {
      ASSERT_FAIL();
    }
  }
}


CTEST2(sort_idx, par_sort)
{
  idx_t * test = splatt_malloc(data->N * sizeof(*test));

  /* make a copy */
  memcpy(test, data->rand_idx, data->N * sizeof(*test));
  /* sort */
  omp_set_num_threads(1);
  quicksort(data->rand_idx, data->N);
  omp_set_num_threads(23);
  quicksort(test, data->N);
  /* compare */
  for(idx_t x=0; x < data->N; ++x) {
    ASSERT_EQUAL(data->rand_idx[x], test[x]);
  }

  /* make a copy */
  memcpy(test, data->fororder, data->N * sizeof(*test));
  /* sort */
  omp_set_num_threads(1);
  quicksort(data->fororder, data->N);
  omp_set_num_threads(23);
  quicksort(test, data->N);
  /* compare */
  for(idx_t x=0; x < data->N; ++x) {
    ASSERT_EQUAL(data->fororder[x], test[x]);
  }

  /* make a copy */
  memcpy(test, data->revorder, data->N * sizeof(*test));
  /* sort */
  omp_set_num_threads(1);
  quicksort(data->revorder, data->N);
  omp_set_num_threads(23);
  quicksort(test, data->N);
  /* compare */
  for(idx_t x=0; x < data->N; ++x) {
    ASSERT_EQUAL(data->revorder[x], test[x]);
  }

  splatt_free(test);
}
