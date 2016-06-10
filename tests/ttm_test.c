
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
  thd_info * thds =  tucker_alloc_thds(opts[SPLATT_OPTION_NTHREADS], csf,
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

  thd_info * thds =  tucker_alloc_thds(opts[SPLATT_OPTION_NTHREADS], csf,
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
          return;
        }
      }
      break;

    case SPLATT_CSF_ONEMODE:
      memcpy(perm, csf[0].dim_perm, nmodes * sizeof(*perm));
      for(idx_t mtest=1; mtest < nmodes; ++mtest) {
        /* skip this if out of order */
        if(perm[mtest] < perm[mtest-1]) {
          return;
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
          return;
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

  csf_free(csf, opts);
  thd_free(thds, opts[SPLATT_OPTION_NTHREADS]);
  free(gold);
  free(test);
}


CTEST_DATA(ttm)
{
  idx_t ntensors;
  idx_t nfactors[MAX_NMODES];
  sptensor_t * tensors[MAX_DSETS];
  matrix_t * mats[MAX_DSETS][MAX_NMODES];
};


CTEST_SETUP(ttm)
{
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
  for(idx_t i=0; i < data->ntensors; ++i) {
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


CTEST2(ttm, csf_one_notile_3mode)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];

    if(tt->nmodes == 3) {
      p_csf_ttm(opts, tt, data->mats[i], data->nfactors);
    }
  }
}


CTEST2(ttm, csf_two_notile_3mode)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_TWOMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];

    if(tt->nmodes == 3) {
      p_csf_ttm(opts, tt, data->mats[i], data->nfactors);
    }
  }
}


CTEST2(ttm, csf_all_notile)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ALLMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];
    p_csf_ttm(opts, tt, data->mats[i], data->nfactors);
  }
}


CTEST2(ttm, rearrange_core_one)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];

    if(tt->nmodes == 3) {
      p_csf_core(opts, tt, data->mats[i], data->nfactors);
    }
  }
}


CTEST2(ttm, rearrange_core_two)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_TWOMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];

    if(tt->nmodes == 3) {
      p_csf_core(opts, tt, data->mats[i], data->nfactors);
    }
  }
}


CTEST2(ttm, rearrange_core_all)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ALLMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];
    p_csf_core(opts, tt, data->mats[i], data->nfactors);
  }
}


