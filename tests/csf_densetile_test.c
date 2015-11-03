#include "../src/csf.h"
#include "../src/tile.h"
#include "../src/sptensor.h"

#include "ctest/ctest.h"
#include "splatt_test.h"


CTEST_DATA(csf_densetile)
{
  sptensor_t * tt;
};

CTEST_SETUP(csf_densetile)
{
  data->tt = tt_read(DATASET(med4.tns));
}

CTEST_TEARDOWN(csf_densetile)
{
  tt_free(data->tt);
}

CTEST2(csf_densetile, csf_one_fill5)
{
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE]      = SPLATT_DENSETILE;
  opts[SPLATT_OPTION_NTHREADS]  = 5;

  splatt_csf * cs = csf_alloc(data->tt, opts);

  ASSERT_EQUAL(data->tt->nnz, cs->nnz);
  ASSERT_EQUAL(data->tt->nmodes, cs->nmodes);

  /* make sure nnz in all tiles adds */
  idx_t countnnz = 0;
  for(idx_t t=0; t < cs->ntiles; ++t) {
    countnnz += cs->pt[t].nfibs[cs->nmodes-1];
  }
  ASSERT_EQUAL(cs->nnz, countnnz);

  /* compare vals and inds */
  idx_t * nnzptr = tt_densetile(data->tt, cs->tile_dims);
  for(idx_t t=0; t < cs->ntiles; ++t) {
    ASSERT_EQUAL(nnzptr[t+1] - nnzptr[t], cs->pt[t].nfibs[cs->nmodes-1]);

    val_t const * const ttvals = data->tt->vals;
    val_t const * const csvals = cs->pt[t].vals;
    idx_t const * const ttinds = data->tt->ind[cs->dim_perm[cs->nmodes-1]];
    idx_t const * const csinds = cs->pt[t].fids[cs->nmodes-1];
    for(idx_t v=nnzptr[t]; v < nnzptr[t+1]; ++v) {
      ASSERT_DBL_NEAR_TOL((double)ttvals[v], (double) csvals[v-nnzptr[t]], 0);
      ASSERT_EQUAL(ttinds[v], csinds[v-nnzptr[t]]);
    }
  }

  free(nnzptr);
  csf_free(cs, opts);
  free(opts);
}


CTEST2(csf_densetile, csf_one_fill1)
{
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE]      = SPLATT_DENSETILE;
  opts[SPLATT_OPTION_NTHREADS]  = 1;

  idx_t ntiles = 1;
  for(idx_t m=0; m < data->tt->nmodes; ++m) {
    ntiles *= opts[SPLATT_OPTION_NTHREADS];
  }

  splatt_csf * cs = csf_alloc(data->tt, opts);

  ASSERT_EQUAL(ntiles, cs->ntiles);
  ASSERT_EQUAL(data->tt->nnz, cs->nnz);
  ASSERT_EQUAL(data->tt->nmodes, cs->nmodes);

  /* make sure nnz in all tiles adds */
  idx_t countnnz = 0;
  for(idx_t t=0; t < cs->ntiles; ++t) {
    countnnz += cs->pt[t].nfibs[cs->nmodes-1];
  }
  ASSERT_EQUAL(cs->nnz, countnnz);

  /* compare vals and inds */
  idx_t * nnzptr = tt_densetile(data->tt, cs->tile_dims);
  for(idx_t t=0; t < cs->ntiles; ++t) {
    ASSERT_EQUAL(nnzptr[t+1] - nnzptr[t], cs->pt[t].nfibs[cs->nmodes-1]);

    val_t const * const ttvals = data->tt->vals;
    val_t const * const csvals = cs->pt[t].vals;
    idx_t const * const ttinds = data->tt->ind[cs->dim_perm[cs->nmodes-1]];
    idx_t const * const csinds = cs->pt[t].fids[cs->nmodes-1];
    for(idx_t v=nnzptr[t]; v < nnzptr[t+1]; ++v) {
      ASSERT_DBL_NEAR_TOL((double)ttvals[v], (double) csvals[v-nnzptr[t]], 0);
      ASSERT_EQUAL(ttinds[v], csinds[v-nnzptr[t]]);
    }
  }

  free(nnzptr);
  csf_free(cs, opts);
  free(opts);
}
