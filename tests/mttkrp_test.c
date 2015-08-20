
#include "../src/mttkrp.h"
#include "../src/ftensor.h"
#include "../src/csf.h"
#include "../src/thd_info.h"

#include "ctest/ctest.h"

#include "splatt_test.h"
#include <omp.h>


static void __compare_mats(
  matrix_t const * const A,
  matrix_t const * const B)
{
  idx_t const nrows = A->I;
  idx_t const ncols = A->J;
  ASSERT_EQUAL(nrows, B->I);
  ASSERT_EQUAL(ncols, B->J);

  for(idx_t i=0; i < A->I; ++i) {
    for(idx_t j=0; j < A->J; ++j) {
      ASSERT_DBL_NEAR_TOL(A->vals[j+(i*ncols)], B->vals[j+(i*ncols)], 1e-10);
    }
  }
}


CTEST_DATA(mttkrp)
{
  idx_t ntensors;
  idx_t nfactors;
  sptensor_t * tensors[MAX_DSETS];
  matrix_t * mats[MAX_DSETS][MAX_NMODES+1];
  matrix_t * gold[MAX_DSETS];
};


CTEST_SETUP(mttkrp)
{
  data->nfactors = 13;
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t const * const tt = data->tensors[i];
    idx_t maxdim = 0;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      data->mats[i][m] = mat_rand(tt->dims[m], data->nfactors);
      maxdim = SS_MAX(tt->dims[m], maxdim);
    }
    data->mats[i][MAX_NMODES] = mat_alloc(maxdim, data->nfactors);
    data->gold[i] = mat_alloc(maxdim, data->nfactors);
  }
}

CTEST_TEARDOWN(mttkrp)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    for(idx_t m=0; m < data->tensors[i]->nmodes; ++m) {
      mat_free(data->mats[i][m]);
    }
    mat_free(data->mats[i][MAX_NMODES]);
    mat_free(data->gold[i]);
    tt_free(data->tensors[i]);
  }
}

CTEST2(mttkrp, splatt)
{
  idx_t const nthreads = 7;
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 1,
    (data->nfactors * data->nfactors * sizeof(val_t)) + 64);

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * const tt = data->tensors[i];
    if(tt->nmodes > 3) {
      continue;
    }
    for(idx_t m=0; m < tt->nmodes; ++m) {
      data->mats[i][MAX_NMODES]->I = tt->dims[m];
      data->gold[i]->I = tt->dims[m];

      /* compute gold */
      mttkrp_stream(tt, data->mats[i], m);

      /* swap to gold */
      matrix_t * tmp = data->mats[i][MAX_NMODES];
      data->mats[i][MAX_NMODES] = data->gold[i];
      data->gold[i] = tmp;

      /* compute splatt */
      ftensor_t ft;
      ften_alloc(&ft, tt, m, SPLATT_NOTILE);
      mttkrp_splatt(&ft, data->mats[i], m, thds, nthreads);
      ften_free(&ft);

      __compare_mats(data->mats[i][MAX_NMODES], data->gold[i]);
    }
  }
  thd_free(thds, nthreads);
}

CTEST2_SKIP(mttkrp, csf)
{
  idx_t const nthreads = 7;
  omp_set_num_threads(nthreads);

  double * opts = splatt_default_opts();

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * const tt = data->tensors[i];
    ctensor_t cs;
    ctensor_alloc(&cs, tt, opts);

    /* add 64 bytes to avoid false sharing */
    thd_info * thds = thd_init(nthreads, 3,
      (data->nfactors * data->nfactors * sizeof(val_t)) + 64,
      0,
      (tt->nmodes * data->nfactors * sizeof(val_t)) + 64);

    for(idx_t m=0; m < tt->nmodes; ++m) {
      data->mats[i][MAX_NMODES]->I = tt->dims[m];
      data->gold[i]->I = tt->dims[m];

      /* compute gold */
      mttkrp_stream(tt, data->mats[i], m);

      /* swap to gold */
      matrix_t * tmp = data->mats[i][MAX_NMODES];
      data->mats[i][MAX_NMODES] = data->gold[i];
      data->gold[i] = tmp;

      /* compute splatt */
      //mttkrp_csf(&cs, data->mats[i], m, thds, nthreads);

      __compare_mats(data->mats[i][MAX_NMODES], data->gold[i]);
    }
    thd_free(thds, nthreads);
    ctensor_free(&cs);
  }
}

