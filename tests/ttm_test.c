
#include "../src/ttm.h"
#include "../src/csf.h"
#include "../src/thd_info.h"
#include "../src/io.h"
#include "../src/util.h"

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
    ASSERT_DBL_NEAR_TOL(gold[i], test[i], 1e-10);
  }
}


static void p_csf_ttm(
    double const * const opts,
    sptensor_t * tt,
    matrix_t ** mats,
    idx_t const * const nfactors)
{
  thd_info * thds = thd_init(opts[SPLATT_OPTION_NTHREADS], 1,
      nfactors[argmax_elem(nfactors, tt->nmodes)] * sizeof(val_t) + 64);

  /* tenout allocations */
  idx_t const outdim = tenout_dim(tt->nmodes, nfactors, tt->dims);
  val_t * gold = calloc(outdim , sizeof(*gold));
  val_t * test = calloc(outdim , sizeof(*test));

  splatt_csf * csf = csf_alloc(tt, opts);

  for(idx_t m=0; m < tt->nmodes; ++m) {
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
    data->nfactors[m] = 3+m;
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


CTEST2(ttm, csf_all_notile_3mode)
{
  idx_t const nthreads = 7;

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_NTHREADS]   = 7;
  opts[SPLATT_OPTION_CSF_ALLOC]  = SPLATT_CSF_ALLMODE;
  opts[SPLATT_OPTION_TILE]       = SPLATT_NOTILE;

  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * tt = data->tensors[i];

    if(tt->nmodes == 3) {
      p_csf_ttm(opts, tt, data->mats[i], data->nfactors);
    }
  }
}



