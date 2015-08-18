
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

  csf_t cs;
  csf_alloc(&cs, data->tt, opts);

  //csf_free(&cs);
}

