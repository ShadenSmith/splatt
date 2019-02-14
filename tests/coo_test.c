
#include "../src/coo.h"

#include "ctest/ctest.h"

#include "splatt_test.h"


CTEST_DATA(sptensor)
{
  idx_t ntensors;
  splatt_coo * tensors[MAX_DSETS];
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
    splatt_coo * gold = data->tensors[i];

    /* make a copy */
    splatt_coo test;
    tt_fill(&test, gold->nnz, gold->nmodes, gold->ind, gold->vals);

    ASSERT_EQUAL(gold->nnz, test.nnz);
    ASSERT_EQUAL(gold->nmodes, test.nmodes);
    for(idx_t m=0; m < gold->nmodes; ++m) {
      ASSERT_EQUAL(gold->dims[m], test.dims[m]);
    }

    splatt_free(test.dims);
  }
}

