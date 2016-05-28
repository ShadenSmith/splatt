
#include "../src/io.h"

#include "ctest/ctest.h"

#include "splatt_test.h"

static char const * const TMP_FILE = "tmp.bin";


CTEST_DATA(io)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
};

CTEST_SETUP(io)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}


CTEST_TEARDOWN(io)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}



CTEST2(io, zero_index)
{
  sptensor_t * zero_tt = tt_read(DATASET(small4.tns));
  sptensor_t * one_tt  = tt_read(DATASET(small4_zeroidx.tns));

  ASSERT_EQUAL(one_tt->nnz, zero_tt->nnz);
  ASSERT_EQUAL(one_tt->nmodes, zero_tt->nmodes);

  for(idx_t m=0; m < one_tt->nmodes; ++m) {
    ASSERT_EQUAL(one_tt->dims[m], zero_tt->dims[m]);

    for(idx_t n=0; n < one_tt->nnz; ++n) {
      ASSERT_EQUAL(one_tt->ind[m][n], zero_tt->ind[m][n]);
    }
  }

  tt_free(zero_tt);
  tt_free(one_tt);
}




CTEST2(io, binary_io)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * const gold = data->tensors[i];

    /* write to binary */
    tt_write_binary(gold, TMP_FILE);

    /* now read it back */
    sptensor_t * tt_bin = tt_read(TMP_FILE);

    /* now check for correctness */
    ASSERT_EQUAL(gold->nnz, tt_bin->nnz);
    ASSERT_EQUAL(gold->nmodes, tt_bin->nmodes);
    for(idx_t m=0; m < tt_bin->nmodes; ++m) {
      idx_t const * const gold_ind = gold->ind[m];
      idx_t const * const test_ind = tt_bin->ind[m];

      for(idx_t n=0; n < tt_bin->nnz; ++n) {
        ASSERT_EQUAL(gold_ind[n], test_ind[n]);
      }
    }

    /* values better be exact! */
    val_t const * const gold_vals = gold->vals;
    val_t const * const test_vals = tt_bin->vals;
    for(idx_t n=0; n < tt_bin->nnz; ++n) {
      ASSERT_DBL_NEAR_TOL(gold_vals[n], test_vals[n], 0.);
    }
  }

  /* delete temporary file */
  remove(TMP_FILE);
}
