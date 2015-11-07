#include "../src/csf.h"
#include "../src/sptensor.h"

#include "ctest/ctest.h"
#include "splatt_test.h"


CTEST_DATA(csf_one_init)
{
  sptensor_t * tt;
  double * opts;
};

CTEST_SETUP(csf_one_init)
{
  data->tt = tt_read(DATASET(med4.tns));

  /* setup opts */
  data->opts = splatt_default_opts();
  data->opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  data->opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  data->opts[SPLATT_OPTION_NTHREADS] = 1;
}

CTEST_TEARDOWN(csf_one_init)
{
  tt_free(data->tt);
  free(data->opts);
}

CTEST2(csf_one_init, fill)
{
  splatt_csf * cs = csf_alloc(data->tt, data->opts);

  ASSERT_EQUAL(1, cs->ntiles);
  ASSERT_EQUAL(data->tt->nnz, cs->nnz);
  ASSERT_EQUAL(data->tt->nmodes, cs->nmodes);

  for(idx_t m=0; m < cs->nmodes; ++m) {
    ASSERT_EQUAL(data->tt->dims[m], cs->dims[m]);
    ASSERT_EQUAL(1, cs->tile_dims[m]);
  }

  /* nfibs[0] should be outer slice dimension */
  ASSERT_EQUAL(data->tt->dims[cs->dim_perm[0]], cs->pt->nfibs[0]);

  /* with 1 tile, vals and inds should be exactly the same ordering */
  idx_t const * const restrict ttinds = data->tt->ind[cs->dim_perm[cs->nmodes-1]];
  idx_t const * const restrict csinds = cs->pt->fids[cs->nmodes-1];
  val_t const * const restrict ttvals = data->tt->vals;
  val_t const * const restrict csvals = cs->pt->vals;
  for(idx_t n=0; n < data->tt->nnz; ++n) {
    ASSERT_DBL_NEAR_TOL((double)ttvals[n], (double) csvals[n], 0);
    ASSERT_EQUAL(ttinds[n], csinds[n]);
  }

  /* no fids needed for root if untiled */
  ASSERT_NULL(cs->pt->fids[0]);

  csf_free(cs, data->opts);
}

CTEST2(csf_one_init, mode_order_small)
{
  splatt_csf * cs = csf_alloc(data->tt, data->opts);

  idx_t * dims = cs->dims;
  idx_t * perm = cs->dim_perm;
  for(idx_t m=0; m < data->tt->nmodes-1; ++m) {
    if(dims[perm[m]] > dims[perm[m+1]]) {
      ASSERT_FAIL();
    }
  }

  csf_free(cs, data->opts);
}


CTEST2(csf_one_init, mode_order_big)
{
  idx_t perm[MAX_NMODES];
  csf_find_mode_order(data->tt->dims, data->tt->nmodes, CSF_SORTED_BIGFIRST, 0,
      perm);

  for(idx_t m=0; m < data->tt->nmodes-1; ++m) {
    if(data->tt->dims[perm[m]] < data->tt->dims[perm[m+1]]) {
      ASSERT_FAIL();
    }
  }
}


CTEST2(csf_one_init, mode_minusone)
{
  idx_t dims[] = {10, 9, 8, 7, 0};
  idx_t perm[] = {0, 0, 0, 0, 0};
  splatt_idx_t ndims = sizeof(dims) / sizeof(dims[0]);

  for(splatt_idx_t m=0; m < ndims; ++m) {
    csf_find_mode_order(dims, ndims, CSF_SORTED_MINUSONE, m, perm);

    if(perm[0] != m) {
      ASSERT_FAIL();
    }
    for(splatt_idx_t m2=2; m2 < ndims-1; ++m2) {
      if(dims[perm[m2]] > dims[perm[m2+1]]) {
        ASSERT_FAIL();
      }
    }
  }
}


CTEST2(csf_one_init, mode_order_inorder)
{
  idx_t dims[] = {10, 9, 8, 7, 0};
  idx_t perm[] = {0, 0, 0, 0, 0};
  splatt_idx_t ndims = sizeof(dims) / sizeof(dims[0]);

  for(splatt_idx_t m=0; m < ndims; ++m) {
    csf_find_mode_order(dims, ndims, CSF_INORDER_MINUSONE, m, perm);

    if(perm[0] != m) {
      ASSERT_FAIL();
    }
    for(splatt_idx_t m2=2; m2 < ndims-1; ++m2) {
      if(perm[m2] > perm[m2+1]) {
        ASSERT_FAIL();
      }
    }
  }
}


CTEST2(csf_one_init, normsq)
{
  val_t gold_norm = 0;
  idx_t nnz = data->tt->nnz;
  val_t const * const vals = data->tt->vals;
  for(idx_t n=0; n < nnz; ++n) {
    gold_norm += vals[n] * vals[n];
  }

  splatt_csf * csf = csf_alloc(data->tt, data->opts);
  val_t mynorm = csf_frobsq(csf);
  csf_free(csf, data->opts);

  ASSERT_DBL_NEAR_TOL(gold_norm, mynorm, 1e-10);
}


CTEST2(csf_one_init, dense_tiled_normsq)
{
  val_t gold_norm = 0;
  idx_t nnz = data->tt->nnz;
  val_t const * const vals = data->tt->vals;
  for(idx_t n=0; n < nnz; ++n) {
    gold_norm += vals[n] * vals[n];
  }

  data->opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
  data->opts[SPLATT_OPTION_NTHREADS] = 7;

  for(idx_t m=0; m < data->tt->nmodes; ++m) {
    data->opts[SPLATT_OPTION_TILEDEPTH] = m;
    splatt_csf * csf = csf_alloc(data->tt, data->opts);
    val_t mynorm = csf_frobsq(csf);
    csf_free(csf, data->opts);

    ASSERT_DBL_NEAR_TOL(gold_norm, mynorm, 1.5e-9);
  }

}
