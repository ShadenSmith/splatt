
#include "ctest/ctest.h"
#include "splatt_test.h"

#include "../src/sptensor.h"


/* API includes */
#include "../include/splatt.h"

#include <omp.h>


CTEST_DATA(api)
{
  splatt_idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
};

CTEST_SETUP(api)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}

CTEST_TEARDOWN(api)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}


CTEST2(api, opts_alloc)
{
  double * opts = splatt_default_opts();
  ASSERT_NOT_NULL(opts);

  /* test defaults */
  ASSERT_EQUAL(omp_get_max_threads(), (int) opts[SPLATT_OPTION_NTHREADS]);

  splatt_free_opts(opts);
}


CTEST2(api, par_opts_alloc)
{
  #pragma omp parallel num_threads(5)
  {
    double * opts = splatt_default_opts();

    ASSERT_EQUAL(1, (int) opts[SPLATT_OPTION_NTHREADS]);
    splatt_free_opts(opts);
  }
}

CTEST2(api, csf_load)
{
  splatt_csf loaded;

  for(idx_t i=0; i < data->ntensors; ++i) {

#if 0
    int ret = splatt_csf_load(datasets[i], &nmodes, &loaded, opts);
#endif
  }
}



