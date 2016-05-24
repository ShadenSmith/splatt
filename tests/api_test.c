
#include "ctest/ctest.h"
#include "splatt_test.h"

#include "../src/sptensor.h"


/* API includes */
#include "../include/splatt.h"

#ifdef _OPENMP
#include <omp.h>
#endif


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


CTEST(api, opts_alloc)
{
  double * opts = splatt_default_opts();
  ASSERT_NOT_NULL(opts);

  /* test defaults */
#ifdef _OPENMP
  ASSERT_EQUAL(omp_get_max_threads(), (int) opts[SPLATT_OPTION_NTHREADS]);
#else
  ASSERT_EQUAL(1, (int) opts[SPLATT_OPTION_NTHREADS]);
#endif

  splatt_free_opts(opts);

  splatt_global_opts * gopts = splatt_alloc_global_opts();
#ifdef _OPENMP
  ASSERT_EQUAL(omp_get_max_threads(), gopts->num_threads);
#else
  ASSERT_EQUAL(1, gopts->num_threads);
#endif
  splatt_free_global_opts(gopts);
}


CTEST(api, par_opts_alloc)
{
  #pragma omp parallel num_threads(5)
  {
    double * opts = splatt_default_opts();
    ASSERT_EQUAL(1, (int) opts[SPLATT_OPTION_NTHREADS]);
    splatt_free_opts(opts);

    splatt_global_opts * gopts = splatt_alloc_global_opts();
    ASSERT_EQUAL(1, gopts->num_threads);
    splatt_free_global_opts(gopts);
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


CTEST(api, version_major)
{
  ASSERT_EQUAL(SPLATT_VER_MAJOR, splatt_version_major());
}

CTEST(api, version_minor)
{
  ASSERT_EQUAL(SPLATT_VER_MINOR, splatt_version_minor());
}

CTEST(api, version_subminor)
{
  ASSERT_EQUAL(SPLATT_VER_SUBMINOR, splatt_version_subminor());
}



