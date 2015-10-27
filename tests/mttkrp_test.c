
#include "../src/mttkrp.h"
#include "../src/ftensor.h"
#include "../src/csf.h"
#include "../src/thd_info.h"

#include "../src/io.h"

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


static void p_csf_mttkrp(
    double const * const opts,
    sptensor_t ** tensors,
    idx_t ntensors,
    matrix_t * mats[][SPLATT_MAX_NMODES+1],
    matrix_t ** gold,
    idx_t nfactors)
{
  idx_t const nthreads = opts[SPLATT_OPTION_NTHREADS];
  for(idx_t i=0; i < ntensors; ++i) {
    sptensor_t * const tt = tensors[i];
    splatt_csf * cs = splatt_csf_alloc(tt, opts);

    /* add 64 bytes to avoid false sharing */
    thd_info * thds = thd_init(nthreads, 3,
      (nfactors * nfactors * sizeof(val_t)) + 64,
      0,
      (tt->nmodes * nfactors * sizeof(val_t)) + 64);

    for(idx_t m=0; m < tt->nmodes; ++m) {
      mats[i][MAX_NMODES]->I = tt->dims[m];
      gold[i]->I = tt->dims[m];

      /* compute gold */
      mttkrp_stream(tt, mats[i], m);

      /* swap to gold */
      matrix_t * tmp = mats[i][MAX_NMODES];
      mats[i][MAX_NMODES] = gold[i];
      gold[i] = tmp;

      /* compute splatt */
      mttkrp_csf(cs, mats[i], m, thds, opts);

      __compare_mats(mats[i][MAX_NMODES], gold[i]);
    }
    thd_free(thds, nthreads);
    csf_free(cs, opts);
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
  data->nfactors = 3;
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


/*
 * SPLATT_CSF_ALLMODE
 */
CTEST2(mttkrp, csf_all_notile)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ALLMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;
  opts[SPLATT_OPTION_TILEDEPTH]  = 0;

  p_csf_mttkrp(opts, data->tensors, data->ntensors, data->mats, data->gold,
      data->nfactors);
}


CTEST2(mttkrp, csf_all_densetile)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ALLMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_DENSETILE;
  opts[SPLATT_OPTION_TILEDEPTH]  = 0;

  p_csf_mttkrp(opts, data->tensors, data->ntensors, data->mats, data->gold,
      data->nfactors);
}


CTEST2(mttkrp, csf_all_densetile_alldepth)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ALLMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_DENSETILE;

  for(int i=1; i < 5; ++i) {
    opts[SPLATT_OPTION_TILEDEPTH]  = i;
    p_csf_mttkrp(opts, data->tensors, data->ntensors, data->mats, data->gold,
        data->nfactors);
  }
}


/*
 * SPLATT_CSF_ONEMODE
 */
CTEST2(mttkrp, csf_one_notile)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;
  opts[SPLATT_OPTION_TILEDEPTH]  = 0;

  p_csf_mttkrp(opts, data->tensors, data->ntensors, data->mats, data->gold,
      data->nfactors);
}


CTEST2(mttkrp, csf_one_densetile)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_DENSETILE;
  opts[SPLATT_OPTION_TILEDEPTH]  = 0;

  p_csf_mttkrp(opts, data->tensors, data->ntensors, data->mats, data->gold,
      data->nfactors);
}


CTEST2_SKIP(mttkrp, csf_one_densetile_alldepth)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_DENSETILE;

  for(int i=1; i < 5; ++i) {
    opts[SPLATT_OPTION_TILEDEPTH]  = i;
    p_csf_mttkrp(opts, data->tensors, data->ntensors, data->mats, data->gold,
        data->nfactors);
  }
}

/*
 * SPLATT_CSF_TWOMODE
 */
CTEST2_SKIP(mttkrp, csf_two_notile)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_TWOMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;
  opts[SPLATT_OPTION_TILEDEPTH]  = 0;

  p_csf_mttkrp(opts, data->tensors, data->ntensors, data->mats, data->gold,
      data->nfactors);
}


CTEST2_SKIP(mttkrp, csf_two_densetile)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_TWOMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_DENSETILE;
  opts[SPLATT_OPTION_TILEDEPTH]  = 0;

  p_csf_mttkrp(opts, data->tensors, data->ntensors, data->mats, data->gold,
      data->nfactors);
}


CTEST2_SKIP(mttkrp, csf_two_densetile_alldepth)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_TWOMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_DENSETILE;

  for(int i=1; i < 5; ++i) {
    opts[SPLATT_OPTION_TILEDEPTH]  = i;
    p_csf_mttkrp(opts, data->tensors, data->ntensors, data->mats, data->gold,
        data->nfactors);
  }
}

