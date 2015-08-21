
#include "../src/csf.h"
#include "../src/tile.h"
#include "../src/sptensor.h"

#include "ctest/ctest.h"
#include "splatt_test.h"


CTEST_DATA(csf_densetile)
{
  sptensor_t * tt;
  idx_t ntiles;
  idx_t tile_dims[MAX_NMODES];
};

CTEST_SETUP(csf_densetile)
{
  data->tt = tt_read(DATASET(med4.tns));
  data->ntiles = 1;
  for(idx_t m=0; m < data->tt->nmodes; ++m) {
    data->tile_dims[m] = 4;
    data->ntiles *= data->tile_dims[m];
  }
}

CTEST_TEARDOWN(csf_densetile)
{
  tt_free(data->tt);
}

CTEST2(csf_densetile, fill)
{
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
  opts[SPLATT_OPTION_NTHREADS] = 4;

  ctensor_t cs;
  ctensor_alloc(&cs, data->tt, opts);

  ASSERT_EQUAL(data->ntiles, cs.ntiles);
  ASSERT_EQUAL(data->tt->nnz, cs.nnz);
  ASSERT_EQUAL(data->tt->nmodes, cs.nmodes);

  /* make sure nnz in all tiles adds */
  idx_t countnnz = 0;
  for(idx_t t=0; t < cs.ntiles; ++t) {
    countnnz += cs.pt[t].nfibs[cs.nmodes-1];
  }
  ASSERT_EQUAL(cs.nnz, countnnz);

  /* compare vals */
  idx_t * nnzptr = tt_densetile(data->tt, cs.tile_dims);
  for(idx_t t=0; t < cs.ntiles; ++t) {
    ASSERT_EQUAL(nnzptr[t+1] - nnzptr[t], cs.pt[t].nfibs[cs.nmodes-1]);

    val_t const * const restrict ttvals = data->tt->vals;
    val_t const * const restrict csvals = cs.pt[t].vals;
    for(idx_t v=nnzptr[t]; v < nnzptr[t+1]; ++v) {
      ASSERT_DBL_NEAR_TOL((double)ttvals[v], (double) csvals[v-nnzptr[t]], 0);
    }
  }

  free(nnzptr);
  ctensor_free(&cs);
  free(opts);
}

