
#include "../src/csf.h"
#include "ctest/ctest.h"
#include "splatt_test.h"


CTEST_DATA(rmse)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
};


CTEST_SETUP(rmse)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}

CTEST_TEARDOWN(rmse)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}


CTEST2_SKIP(rmse, tensor_1nnz)
{
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_VERBOSITY] = SPLATT_VERBOSITY_NONE;
  opts[SPLATT_OPTION_NITER] = 50;
  opts[SPLATT_OPTION_TOLERANCE] = 1e-8;

  idx_t coords[SPLATT_MAX_NMODES];
  for(idx_t m=0; m < SPLATT_MAX_NMODES; ++m) {
    coords[m] = 0;
  }

  for(idx_t nmodes=3; nmodes < SPLATT_MAX_NMODES; ++nmodes) {
    sptensor_t * tt = tt_alloc(nmodes, nmodes);

    for(idx_t m=0; m < nmodes; ++m) {
      tt->dims[m] = nmodes;
      for(idx_t n=0; n < tt->nnz; ++n) {
        tt->ind[m][n] = n;
      }
    }
    for(idx_t n=0; n < tt->nnz; ++n) {
      tt->vals[n] = 1.38 * (n+1) + n;
    }

    splatt_csf * csf = csf_alloc(tt, opts);

    splatt_kruskal factored;
    splatt_cpd_als(csf, csf->nmodes + 1, opts, &factored);
    splatt_free_csf(csf, opts);

    /* now predict */
    val_t predict;
    splatt_kruskal_predict(&factored, coords, &predict);
    ASSERT_DBL_NEAR_TOL(tt->vals[0], predict, 1e-5);

    splatt_free_kruskal(&factored);
    tt_free(tt);
  }

  splatt_free_opts(opts);
}
