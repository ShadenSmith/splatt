
#include "../src/tile.h"
#include "../src/sptensor.h"

#include "ctest/ctest.h"
#include "splatt_test.h"


CTEST_DATA(tile)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
};


CTEST_SETUP(tile)
{
  data->ntensors = load_tensors(data->tensors);
}


CTEST_TEARDOWN(tile)
{
  free_tensors(data->tensors);
}



/*
 * Use only 1 tile on a tensor with 'MAX_NMODES' modes. id should be 0.
 */
CTEST(tile, get_tile_id_zero)
{
  idx_t dims[MAX_NMODES];
  idx_t coords[MAX_NMODES];
  /* one tile, id should always be 0 */
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    dims[m] = 1;
    coords[m] = 1;
  }
  ASSERT_EQUAL(0, get_tile_id(dims, MAX_NMODES, coords));
}


/*
 * Test get_tile_id on a 3D problem with a prime number of threads.
 */
CTEST(tile, get_tile_id_3d)
{
  idx_t const nmodes = 3;
  idx_t const nthreads = 7;
  idx_t dims[MAX_NMODES];
  idx_t coords[MAX_NMODES];

  for(idx_t m=0; m < nmodes; ++m) {
    dims[m] = nthreads;
  }

  idx_t id = 0;
  for(idx_t m1=0; m1 < nthreads; ++m1) {
    coords[0] = m1+1;
    for(idx_t m2=0; m2 < nthreads; ++m2) {
      coords[1] = m2+1;
      for(idx_t m3=0; m3 < nthreads; ++m3) {
        coords[2] = m3+1;
        ASSERT_EQUAL(id, get_tile_id(dims, nmodes, coords));
        ++id;
      }
    }
  }
}


CTEST(tile, fill_tile_coords)
{
  idx_t const nmodes = 4;
  idx_t const nthreads = 4;
  idx_t coords[MAX_NMODES];
  idx_t dims[MAX_NMODES];
  idx_t ntiles = 1;

  for(idx_t m=0; m < nmodes; ++m) {
    dims[m] = nthreads;
    ntiles *= nthreads;
  }

  for(idx_t t=0; t < ntiles; ++t) {
    fill_tile_coords(dims, nmodes, t, coords);
    ASSERT_EQUAL(t, get_tile_id(dims, nmodes, coords));
  }
}


CTEST(tile, get_tile_id_nd)
{
  idx_t const nmodes = 3;
  idx_t const nthreads = 3;
  idx_t dims[MAX_NMODES];
  idx_t coords[MAX_NMODES];

  for(idx_t m=0; m < nmodes; ++m) {
    dims[m] = nthreads;
  }

  for(idx_t m=0; m < nmodes; ++m) {
    idx_t id = TILE_BEGIN;
    for(idx_t t=0; t < nthreads; ++t) {
      id = get_next_tileid(id, dims, nmodes, m);
      printf("id: %lu\n", id);
      ASSERT_NOT_EQUAL(TILE_END, id);
    }
    id = get_next_tileid(id, dims, nmodes, m);
    ASSERT_EQUAL(TILE_END, id);
  }
}


