
#include "../src/tile.h"
#include "../src/coo.h"

#include "ctest/ctest.h"
#include "splatt_test.h"


CTEST_DATA(tile_dense)
{
  splatt_coo * tt;
  idx_t ntiles;
  idx_t tile_dims[MAX_NMODES];
};

CTEST_SETUP(tile_dense)
{
  data->tt = tt_read(DATASET(med4.tns));
  data->ntiles = 1;
  for(idx_t m=0; m < data->tt->nmodes; ++m) {
    data->tile_dims[m] = 4;
    data->ntiles *= data->tile_dims[m];
  }
}

CTEST_TEARDOWN(tile_dense)
{
  tt_free(data->tt);
}


/*
 * Use a basic checksum to ensure  no nnz went missing after tiling. A pass
 * does not rule out a false negative!
 */
CTEST2(tile_dense, no_missing_nnz)
{
  idx_t cksums[MAX_NMODES];
  for(idx_t m=0; m < data->tt->nmodes; ++m) {
    cksums[m] = 0;

    splatt_coo const * const tt = data->tt;
    for(idx_t x=0; x < tt->nnz; ++x) {
      cksums[m] += tt->ind[m][x];
    }
  }
  double valsum = 0;
  for(idx_t x=0; x < data->tt->nnz; ++x) {
    valsum += data->tt->vals[x];
  }

  idx_t * ptr = tt_densetile(data->tt, data->tile_dims);
  ASSERT_EQUAL(data->tt->nnz, ptr[data->ntiles]);
  free(ptr);

  for(idx_t m=0; m < data->tt->nmodes; ++m) {
    splatt_coo const * const tt = data->tt;
    for(idx_t x=0; x < tt->nnz; ++x) {
      cksums[m] -= tt->ind[m][x];
    }

    ASSERT_EQUAL(0, cksums[m]);
  }

  double valsum2 = 0;
  for(idx_t x=0; x < data->tt->nnz; ++x) {
    valsum2 += data->tt->vals[x];
  }

  ASSERT_DBL_NEAR_TOL(valsum, valsum2, 1.5e-9);
}


/*
 * Use a basic checksum to ensure  no nnz went missing after tiling. We use
 * the tile ptr to traverse the tensor and check. A pass does not rule out a
 * false negative!
 */
CTEST2(tile_dense, no_missing_nnz_traverse)
{
  idx_t cksums[MAX_NMODES];
  for(idx_t m=0; m < data->tt->nmodes; ++m) {
    cksums[m] = 0;

    splatt_coo const * const tt = data->tt;
    for(idx_t x=0; x < tt->nnz; ++x) {
      cksums[m] += tt->ind[m][x];
    }
  }

  idx_t * ptr = tt_densetile(data->tt, data->tile_dims);
  ASSERT_EQUAL(data->tt->nnz, ptr[data->ntiles]);

  /* use get_next_tileid to simulate traversal */
  for(idx_t i=0; i < data->tile_dims[0]; ++i) {
    idx_t id;
    id = get_next_tileid(TILE_BEGIN, data->tile_dims, data->tt->nmodes, 0, i);
    while(id != TILE_END) {
      idx_t const start = ptr[id];
      idx_t const end = ptr[id+1];
      for(idx_t m=0; m < data->tt->nmodes; ++m) {
        splatt_coo const * const tt = data->tt;
        for(idx_t x=start; x < end; ++x) {
          cksums[m] -= tt->ind[m][x];
        }
      }

      /* next tile */
      id = get_next_tileid(id, data->tile_dims, data->tt->nmodes, 0, i);
    }
  }

  /* hopefully we ended up at 0... */
  for(idx_t m=0; m < data->tt->nmodes; ++m) {
    ASSERT_EQUAL(0, cksums[m]);
  }
  free(ptr);
}


/*
 * Perform a dense tiling and ensure every nnz has the index range that is
 * expected.
 */
CTEST2(tile_dense, check_tile_bounds)
{
  idx_t * ptr = tt_densetile(data->tt, data->tile_dims);

  idx_t coords[MAX_NMODES];

  splatt_coo const * const tt = data->tt;
  for(idx_t i=0; i < data->tile_dims[0]; ++i) {
    idx_t id;
    id = get_next_tileid(TILE_BEGIN, data->tile_dims, data->tt->nmodes, 0, i);
    while(id != TILE_END) {
      fill_tile_coords(data->tile_dims, tt->nmodes, id, coords);
      idx_t const startnnz = ptr[id];
      idx_t const endnnz = ptr[id+1];
      for(idx_t x=startnnz; x < endnnz; ++x) {
        for(idx_t m=0; m < tt->nmodes; ++m) {
          idx_t const tsize = tt->dims[m] / data->tile_dims[m];
          idx_t const minidx = coords[m] * tsize;
          idx_t const maxidx = (coords[m] + 1) * tsize;

          ASSERT_EQUAL(1, tt->ind[m][x] < tt->dims[m]);

          /* assert we are in the correct range */
          ASSERT_EQUAL(1, tt->ind[m][x] >= minidx);
          /* last coordinate may have overflow */
          if(coords[m]+1 < data->tile_dims[m]) {
            ASSERT_EQUAL(1, tt->ind[m][x] < maxidx);
          }
        }
      }

      /* next tile */
      id = get_next_tileid(id, data->tile_dims, tt->nmodes, 0, i);
    }
  }
  free(ptr);
}


/*
 * Perform a dense tiling and ensure every nnz has the index range that is
 * expected.
 */
CTEST2(tile_dense, check_tile_bounds_weirddims)
{
  for(idx_t m=0; m < data->tt->nmodes; ++m) {
    data->tile_dims[m] = m+1;
  }
  idx_t * ptr = tt_densetile(data->tt, data->tile_dims);

  idx_t coords[MAX_NMODES];

  splatt_coo const * const tt = data->tt;
  for(idx_t i=0; i < data->tile_dims[0]; ++i) {
    idx_t id;
    id = get_next_tileid(TILE_BEGIN, data->tile_dims, data->tt->nmodes, 0, i);
    while(id != TILE_END) {
      fill_tile_coords(data->tile_dims, tt->nmodes, id, coords);
      idx_t const startnnz = ptr[id];
      idx_t const endnnz = ptr[id+1];
      for(idx_t x=startnnz; x < endnnz; ++x) {
        for(idx_t m=0; m < tt->nmodes; ++m) {
          idx_t const tsize = tt->dims[m] / data->tile_dims[m];
          idx_t const minidx = coords[m] * tsize;
          idx_t const maxidx = (coords[m] + 1) * tsize;

          /* assert we are in the correct range */
          ASSERT_EQUAL(1, tt->ind[m][x] >= minidx);
          /* last coordinate may have overflow */
          if(coords[m]+1 < data->tile_dims[m]) {
            ASSERT_EQUAL(1, tt->ind[m][x] < maxidx);
          }
        }
      }

      /* next tile */
      id = get_next_tileid(id, data->tile_dims, tt->nmodes, 0, i);
    }
  }
  free(ptr);
}

