
#include "../src/sptensor.h"
#include "../src/io.h"
#include "../src/sort.h"
#include "../src/tile.h"

#include "ctest/ctest.h"

#include "splatt_test.h"
#include <omp.h>


/**
* @brief Return 1 if inds[mode][a] <= inds[mode][b].
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

  for(idx_t m=0; m < nmodes; ++m) {
    if(m == mode) {
      continue;
    }

    if(inds[m][a] < inds[m][b]) {
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
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];
    for(idx_t m=tt->nmodes; m-- != 0; ) {
      tt_sort(tt, m, NULL);

      for(idx_t x=0; x < tt->nnz-1; ++x) {
        ASSERT_EQUAL(1, __idx_cmp(tt->ind, m, tt->nmodes, x, x+1));
      }
    }
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


