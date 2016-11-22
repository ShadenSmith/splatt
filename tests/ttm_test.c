
#include "../src/ttm.h"
#include "../src/csf.h"
#include "../src/thd_info.h"
#include "../src/io.h"
#include "../src/util.h"
#include "../src/svd.h"
#include "../src/tucker.h"

#include "ctest/ctest.h"

#include "splatt_test.h"
#include <omp.h>


/**
* @brief Compare two vectors for (near-)equality.
*
* @param test The one we are testing.
* @param gold The known correct one.
* @param len_test The length of the test vector.
* @param len_gold The length of the gold vector.
*/
static void p_compare_vecs(
    val_t const * const test,
    val_t const * const gold,
    idx_t const len_test,
    idx_t const len_gold)
{
  ASSERT_EQUAL(len_gold, len_test);

  for(idx_t i=0; i < len_gold; ++i) {
    ASSERT_DBL_NEAR_TOL(gold[i], test[i], 1e-9);
  }
}


static void p_csf_core(
    double const * const opts,
    sptensor_t * tt,
    matrix_t ** mats,
    idx_t const * const nfactors)
{
  idx_t const nmodes = tt->nmodes;

  /* tenout allocations */
  idx_t const outdim = tenout_dim(nmodes, nfactors, tt->dims);
  val_t * gold_ttmc = calloc(outdim , sizeof(*gold_ttmc));
  val_t * test_ttmc = calloc(outdim , sizeof(*test_ttmc));

  /* core allocations */
  idx_t core_size = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    core_size *= nfactors[m];
  }
  val_t * gold_core = calloc(core_size, sizeof(*gold_core));
  val_t * test_core = calloc(core_size, sizeof(*test_core));

  splatt_csf * csf = csf_alloc(tt, opts);
  thd_info * thds =  ttmc_alloc_thds(opts[SPLATT_OPTION_NTHREADS], csf,
      nfactors, opts);

  /* make a matrix out of TTMc */
  matrix_t gold_mat;
  gold_mat.I = tt->dims[0];
  gold_mat.J = 1;
  for(idx_t m=1; m < tt->nmodes; ++m) {
    gold_mat.J *= nfactors[m];
  }
  gold_mat.vals = gold_ttmc;

  matrix_t core_mat;
  core_mat.I = nfactors[0];
  core_mat.J = core_size / nfactors[0];
  core_mat.vals = gold_core;

  /* compute gold, always with first mode for core ordering */
  ttmc_stream(tt, mats, gold_ttmc, 0, opts);
  matrix_t * A = mat_alloc(mats[0]->I, nfactors[0]);
  mat_transpose(mats[0], A);
  mat_matmul(A, &gold_mat, &core_mat);
  mat_free(A);

  /* compute CSF test */
  ttmc_csf(csf, mats, test_ttmc, nmodes-1, thds, opts);
  make_core(test_ttmc, mats[nmodes-1]->vals, test_core, nmodes, nmodes-1,
      nfactors, csf->dims[nmodes-1]);
  permute_core(csf, test_core, nfactors, opts);

  /* compare */
  p_compare_vecs(test_core, gold_core, core_size, core_size);

  csf_free(csf, opts);
  thd_free(thds, opts[SPLATT_OPTION_NTHREADS]);
  free(gold_ttmc);
  free(test_ttmc);
}


static void p_csf_ttm(
    double const * const opts,
    sptensor_t * tt,
    matrix_t ** mats,
    idx_t const * const nfactors)
{
  idx_t const nmodes = tt->nmodes;

  /* tenout allocations */
  idx_t const outdim = tenout_dim(nmodes, nfactors, tt->dims);
  val_t * gold = calloc(outdim , sizeof(*gold));
  val_t * test = calloc(outdim , sizeof(*test));

  splatt_csf * csf = csf_alloc(tt, opts);
  idx_t perm[MAX_NMODES];

  thd_info * thds =  ttmc_alloc_thds(opts[SPLATT_OPTION_NTHREADS], csf,
      nfactors, opts);

  for(idx_t m=0; m < nmodes; ++m) {
    /* XXX only test when dim_perm is sorted, because columns in tenout will be
     * permuted otherwise */
    splatt_csf_type const which = opts[SPLATT_OPTION_CSF_ALLOC];
    switch(which) {
    case SPLATT_CSF_ALLMODE:
      memcpy(perm, csf[m].dim_perm, nmodes * sizeof(*perm));
      for(idx_t mtest=2; mtest < nmodes; ++mtest) {
        /* skip if out of order */
        if(perm[mtest] < perm[mtest-1]) {
          goto CLEANUP;
        }
      }
      break;

    case SPLATT_CSF_ONEMODE:
      memcpy(perm, csf[0].dim_perm, nmodes * sizeof(*perm));
      for(idx_t mtest=1; mtest < nmodes; ++mtest) {
        /* skip this if out of order */
        if(perm[mtest] < perm[mtest-1]) {
          goto CLEANUP;
        }
      }
      break;

    case SPLATT_CSF_TWOMODE:
      if(m == csf[0].dim_perm[nmodes-1]) {
        memcpy(perm, csf[1].dim_perm, nmodes * sizeof(*perm));
      } else {
        memcpy(perm, csf[0].dim_perm, nmodes * sizeof(*perm));
      }
      for(idx_t mtest=1; mtest < nmodes; ++mtest) {
        /* skip this if out of order */
        if(perm[mtest] < perm[mtest-1]) {
          goto CLEANUP;
        }
      }
      break;
    }

    /* compute gold */
    ttmc_stream(tt, mats, gold, m, opts);

    /* compute CSF test */
    ttmc_csf(csf, mats, test, m, thds, opts);

    /* compare */
    p_compare_vecs(test, gold, outdim, outdim);
  }

  CLEANUP:
  csf_free(csf, opts);
  thd_free(thds, opts[SPLATT_OPTION_NTHREADS]);
  free(gold);
  free(test);
}


CTEST_DATA(ttm)
{
  double * opts;
  idx_t ntensors;
  idx_t nfactors[MAX_NMODES];
  sptensor_t * tensors[MAX_DSETS];
  matrix_t * mats[MAX_DSETS][MAX_NMODES];
};


CTEST_SETUP(ttm)
{
  data->opts = splatt_default_opts();
  data->opts[SPLATT_OPTION_NTHREADS] = 7;

  for(idx_t m=0; m < MAX_NMODES; ++m) {
    data->nfactors[m] = 3;//+m;
  }

  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t const * const tt = data->tensors[i];
    for(idx_t m=0; m < tt->nmodes; ++m) {
      data->mats[i][m] = mat_rand(tt->dims[m], data->nfactors[m]);
    }
  }
}


CTEST_TEARDOWN(ttm)
{
  splatt_free_opts(data->opts);

  for(idx_t i=0; i < data->ntensors; ++i) {
    for(idx_t m=0; m < data->tensors[i]->nmodes; ++m) {
      mat_free(data->mats[i][m]);
    }
    tt_free(data->tensors[i]);
  }
}


CTEST2(ttm, tenout_alloc)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t const * const tt = data->tensors[i];
    idx_t const size = tenout_dim(tt->nmodes, data->nfactors, tt->dims);

    /* compute size */
    for(idx_t m=0; m < tt->nmodes; ++m) {
      idx_t const nrows = tt->dims[m];
      idx_t ncols = 1;
      for(idx_t m2=0; m2 < tt->nmodes; ++m2) {
        if(m != m2) {
          ncols *= data->nfactors[m2];
        }
      }

      idx_t const total = nrows * ncols;
      if(total > size) {
        ASSERT_FAIL();
      }
    }
  } /* foreach tensor */
}


CTEST2(ttm, csf_one_notile)
{
  data->opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ONEMODE;
  data->opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];
    p_csf_ttm(data->opts, tt, data->mats[i], data->nfactors);
  }
}


CTEST2(ttm, csf_two_notile)
{
  data->opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_TWOMODE;
  data->opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];
    p_csf_ttm(data->opts, tt, data->mats[i], data->nfactors);
  }
}


CTEST2(ttm, csf_all_notile)
{
  data->opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ALLMODE;
  data->opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];
    p_csf_ttm(data->opts, tt, data->mats[i], data->nfactors);
  }
}


CTEST2(ttm, rearrange_core_one)
{
  data->opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ONEMODE;
  data->opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];

    p_csf_core(data->opts, tt, data->mats[i], data->nfactors);
  }
}


CTEST2(ttm, rearrange_core_two)
{
  data->opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_TWOMODE;
  data->opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];

    p_csf_core(data->opts, tt, data->mats[i], data->nfactors);
  }
}


CTEST2(ttm, rearrange_core_all)
{
  data->opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ALLMODE;
  data->opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];
    p_csf_core(data->opts, tt, data->mats[i], data->nfactors);
  }
}


CTEST2(ttm, ttmc_full)
{
  sptensor_t * tt = tt_alloc(3, 3);
  tt->dims[0] = 3;
  tt->dims[1] = 3;
  tt->dims[2] = 3;

  tt->vals[0] = 1.;
  tt->ind[0][0] = 0;
  tt->ind[1][0] = 0;
  tt->ind[2][0] = 0;

  tt->vals[1] = 2.;
  tt->ind[0][1] = 1;
  tt->ind[1][1] = 1;
  tt->ind[2][1] = 1;

  tt->vals[2] = 3.;
  tt->ind[0][2] = 2;
  tt->ind[1][2] = 2;
  tt->ind[2][2] = 1;

  double * opts = splatt_default_opts();
  splatt_csf * csf = csf_alloc(tt, opts);
  idx_t ncolumns[MAX_NMODES];

  /* factors of all 1 */
  val_t * mats[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    ncolumns[m] = 2;
    mats[m] = splatt_malloc(tt->dims[m] * ncolumns[m] * sizeof(**mats));
    for(idx_t x=0; x < tt->dims[m] * ncolumns[m]; ++x) {
      mats[m][x] = 1.;
    }
  }

  idx_t const core_size = ncolumns[0] * ncolumns[1] * ncolumns[2];
  val_t * core = splatt_malloc(core_size * sizeof(*core));

  int ret = splatt_ttmc_full(ncolumns, csf, mats, core, opts);
  ASSERT_EQUAL(SPLATT_SUCCESS, ret);
  for(idx_t x=0; x < core_size; ++x) {
    ASSERT_DBL_NEAR_TOL(6., core[x], 1e-12);
  }

  tt_free(tt);
  splatt_free(core);
  splatt_free_csf(csf, opts);
  splatt_free_opts(opts);
}

