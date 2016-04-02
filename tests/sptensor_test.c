
#include "../src/sptensor.h"

#include "ctest/ctest.h"

#include "splatt_test.h"
#include <omp.h>


CTEST_DATA(sptensor)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
};


CTEST_SETUP(sptensor)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}

CTEST_TEARDOWN(sptensor)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}


CTEST2(sptensor, tt_fill)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * gold = data->tensors[i];

    /* make a copy */
    sptensor_t test;
    tt_fill(&test, gold->nnz, gold->nmodes, gold->ind, gold->vals);

    ASSERT_EQUAL(gold->nnz, test.nnz);
    ASSERT_EQUAL(gold->nmodes, test.nmodes);
    for(idx_t m=0; m < gold->nmodes; ++m) {
      ASSERT_EQUAL(gold->dims[m], test.dims[m]);
    }

    splatt_free(test.dims);
  }
}


CTEST2(sptensor, tt_copy)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * gold = data->tensors[i];

    sptensor_t * cpy = tt_copy(gold);

    ASSERT_NOT_NULL(cpy);
    ASSERT_EQUAL(gold->nnz, cpy->nnz);
    ASSERT_EQUAL(gold->nmodes, cpy->nmodes);
    ASSERT_EQUAL(gold->type, cpy->type);

    /* ensure it's actually a deep cpy */
    if(gold->vals == cpy->vals) {
      ASSERT_FAIL();
    }

    /* check indices */
    for(idx_t m=0; m < gold->nmodes; ++m) {
      for(idx_t n=0; n < gold->nnz; ++n) {
        ASSERT_EQUAL(gold->ind[m][n], cpy->ind[m][n]);
      }
    }

    /* check values */
    for(idx_t n=0; n < cpy->nnz; ++n) {
      ASSERT_EQUAL(gold->vals[n], cpy->vals[n]);
    }

    ASSERT_EQUAL(gold->tiled, cpy->tiled);

    /* check indmap */
    for(idx_t m=0; m < gold->nmodes; ++m) {
      if(gold->indmap[m] == NULL) {
        ASSERT_NULL(cpy->indmap[m]);
      } else {
        for(idx_t n=0; n < gold->nnz; ++n) {
          ASSERT_EQUAL(gold->indmap[m][n], cpy->indmap[m][n]);
        }
      }
    }

    tt_free(cpy);
  } /* foreach tensor */
}


CTEST2(sptensor, tt_union_copy)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * gold = data->tensors[i];
    sptensor_t * gold2 = tt_copy(gold);

    sptensor_t * unn = tt_union(gold, gold2);
    ASSERT_NOT_NULL(unn);
    ASSERT_EQUAL(gold->nnz, unn->nnz);

    for(idx_t n=0; n < gold->nnz; ++n) {
      ASSERT_EQUAL(gold->vals[n], unn->vals[n]);
    }

    for(idx_t m=0; m < gold->nmodes; ++m) {
      for(idx_t n=0; n < gold->nnz; ++n) {
        ASSERT_EQUAL(gold->ind[m][n], unn->ind[m][n]);
      }
    }

    tt_free(gold2);
    tt_free(unn);
  }
}


CTEST2(sptensor, tt_union_empty)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * gold = data->tensors[i];

    sptensor_t * empty = tt_alloc(0, gold->nmodes);
    sptensor_t * unn = tt_union(gold, empty);

    ASSERT_NOT_NULL(unn);
    ASSERT_EQUAL(gold->nnz, unn->nnz);

    for(idx_t n=0; n < gold->nnz; ++n) {
      ASSERT_EQUAL(gold->vals[n], unn->vals[n]);
    }

    for(idx_t m=0; m < gold->nmodes; ++m) {
      for(idx_t n=0; n < gold->nnz; ++n) {
        ASSERT_EQUAL(gold->ind[m][n], unn->ind[m][n]);
      }
    }
    tt_free(unn);

    /* now swap order */
    unn = tt_union(empty, gold);
    ASSERT_NOT_NULL(unn);
    ASSERT_EQUAL(gold->nnz, unn->nnz);

    for(idx_t n=0; n < gold->nnz; ++n) {
      ASSERT_EQUAL(gold->vals[n], unn->vals[n]);
    }

    for(idx_t m=0; m < gold->nmodes; ++m) {
      for(idx_t n=0; n < gold->nnz; ++n) {
        ASSERT_EQUAL(gold->ind[m][n], unn->ind[m][n]);
      }
    }
    tt_free(unn);

    tt_free(empty);
  }
}

