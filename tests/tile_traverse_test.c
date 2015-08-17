
#include "../src/tile.h"
#include "../src/sptensor.h"

#include "ctest/ctest.h"
#include "splatt_test.h"


static void __fill_arr(
    idx_t * const arr,
    idx_t const len,
    idx_t const val)
{
  for(idx_t m=0; m < len; ++m) {
    arr[m] = val;
  }
}


CTEST_DATA(tile_traverse)
{
  idx_t dims[MAX_NMODES];
  idx_t coords[MAX_NMODES];
};


CTEST_SETUP(tile_traverse)
{
  __fill_arr(data->dims, MAX_NMODES, 0);
  __fill_arr(data->coords, MAX_NMODES, 0);
}


CTEST_TEARDOWN(tile_traverse)
{
}


/*
 * Use only 1 tile on a tensor with 'MAX_NMODES' modes. id should be 0.
 */
CTEST2(tile_traverse, get_tile_id_zero)
{
  __fill_arr(data->dims, MAX_NMODES, 1);
  __fill_arr(data->coords, MAX_NMODES, 0);

  /* one tile_traverse, id should always be 0 */
  ASSERT_EQUAL(0, get_tile_id(data->dims, MAX_NMODES, data->coords));
}


/*
 * Test get_tile_id on a 3D problem with a prime number of threads.
 */
CTEST2(tile_traverse, get_tile_id_3d)
{
  idx_t const nmodes = 3;
  idx_t const nthreads = 7;

  __fill_arr(data->dims, nmodes, nthreads);

  idx_t id = 0;
  for(idx_t m1=0; m1 < nthreads; ++m1) {
    data->coords[0] = m1;
    for(idx_t m2=0; m2 < nthreads; ++m2) {
      data->coords[1] = m2;
      for(idx_t m3=0; m3 < nthreads; ++m3) {
        data->coords[2] = m3;
        ASSERT_EQUAL(id, get_tile_id(data->dims, nmodes, data->coords));
        ++id;
      }
    }
  }
}

/*
 * Test get_tile_id on a 3D problem with weird dimensions.
 */
CTEST2(tile_traverse, get_tile_id_weird_dim)
{
  idx_t const nmodes = 3;

  for(idx_t m=0; m < nmodes; ++m) {
    data->dims[m] = m+1;
  }

  idx_t id = 0;
  for(idx_t m1=0; m1 < data->dims[0]; ++m1) {
    data->coords[0] = m1;
    for(idx_t m2=0; m2 < data->dims[1]; ++m2) {
      data->coords[1] = m2;
      for(idx_t m3=0; m3 < data->dims[2]; ++m3) {
        data->coords[2] = m3;
        ASSERT_EQUAL(id, get_tile_id(data->dims, nmodes, data->coords));
        ++id;
      }
    }
  }
}


/*
 * Test get_tile_id on out of bounds values.
 */
CTEST2(tile_traverse, get_tile_id_oob)
{
  idx_t const nmodes = MAX_NMODES;
  for(idx_t m=0; m < nmodes; ++m) {
    data->dims[m] = m+1;
    data->coords[m] = m+1;
  }

  /* check out of bounds */
  ASSERT_EQUAL(TILE_ERR, get_tile_id(data->dims, nmodes, data->coords));

  for(idx_t m=0; m < nmodes; ++m) {
    data->coords[m] = m;
  }
  ASSERT_NOT_EQUAL(TILE_ERR, get_tile_id(data->dims, nmodes, data->coords));

  /* check each mode individually */
  for(idx_t m=0; m < nmodes; ++m) {
    ++data->coords[m];
    ASSERT_EQUAL(TILE_ERR, get_tile_id(data->dims, nmodes, data->coords));
    --data->coords[m];
  }
}


CTEST2(tile_traverse, fill_tile_coords)
{
  idx_t const nmodes = 4;
  idx_t const nthreads = 4;
  idx_t ntiles = 1;

  for(idx_t m=0; m < nmodes; ++m) {
    data->dims[m] = nthreads;
    ntiles *= nthreads;
  }

  for(idx_t t=0; t < ntiles; ++t) {
    fill_tile_coords(data->dims, nmodes, t, data->coords);
    ASSERT_EQUAL(t, get_tile_id(data->dims, nmodes, data->coords));
  }
}


/* Test invalid values for fill_tile_coords() */
CTEST2(tile_traverse, fill_tile_coords_oob)
{
  idx_t nmodes = MAX_NMODES;
  idx_t ntiles = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    data->dims[m] = m+1;
    ntiles *= data->dims[m];
  }

  /* check max tile */
  fill_tile_coords(data->dims, nmodes, ntiles, data->coords);
  for(idx_t m=0; m < nmodes; ++m) {
    ASSERT_EQUAL(data->dims[m], data->coords[m]);
  }

  /* check other invalids */
  fill_tile_coords(data->dims, nmodes, TILE_BEGIN, data->coords);
  for(idx_t m=0; m < nmodes; ++m) {
    ASSERT_EQUAL(data->dims[m], data->coords[m]);
  }
  fill_tile_coords(data->dims, nmodes, TILE_END, data->coords);
  for(idx_t m=0; m < nmodes; ++m) {
    ASSERT_EQUAL(data->dims[m], data->coords[m]);
  }
  fill_tile_coords(data->dims, nmodes, TILE_ERR, data->coords);
  for(idx_t m=0; m < nmodes; ++m) {
    ASSERT_EQUAL(data->dims[m], data->coords[m]);
  }
}

/*
 * Test TILE_BEGIN functionality.
 */
CTEST2(tile_traverse, get_tile_id_begin)
{
  idx_t const nmodes = 6;
  idx_t const nthreads = 3;

  __fill_arr(data->dims, nmodes, nthreads);

  for(idx_t m=0; m < nmodes; ++m) {
    for(idx_t d=0; d < data->dims[m]; ++d) {
      /* get starting id */
      idx_t const b_id = get_next_tileid(TILE_BEGIN, data->dims, nmodes, m, d);

      /* now do it ourselves */
      __fill_arr(data->coords, nmodes, 0);
      data->coords[m] = d;
      idx_t const c_id = get_tile_id(data->dims, nmodes, data->coords);

      ASSERT_EQUAL(c_id, b_id);
    }
  }
}


/*
 * Test TILE_END functionality.
 */
CTEST2(tile_traverse, get_tile_id_end)
{
  idx_t const nmodes = MAX_NMODES;
  idx_t const nthreads = 3;

  __fill_arr(data->dims, nmodes, nthreads);

  for(idx_t m=0; m < nmodes; ++m) {
    /* very last tile */
    __fill_arr(data->coords, nmodes, nthreads-1);

    /* now ensure every idx in the mode sees that it is the end */
    for(idx_t d=0; d < data->dims[m]; ++d) {
      /* now do it ourselves */
      data->coords[m] = d;
      idx_t const t_id = get_tile_id(data->dims, nmodes, data->coords);
      idx_t const b_id = get_next_tileid(t_id, data->dims, nmodes, m, d);
      ASSERT_EQUAL(TILE_END, b_id);
    }
  }
}


/*
 * Do an entire traversal over a tile space.
 */
CTEST2(tile_traverse, get_tile_id)
{
  idx_t const nmodes = MAX_NMODES;
  idx_t const nthreads = 3;

  __fill_arr(data->dims, nmodes, nthreads);

  for(idx_t m=0; m < nmodes; ++m) {
    /* empty tiles */
    __fill_arr(data->coords, nmodes, 0);

    /* the number of tiles that the traversal should go through */
    idx_t ntiles = 1;
    for(idx_t m2=1; m2 < nmodes; ++m2) {
      ntiles *= data->dims[m2];
    }

    /* now ensure every idx in the mode sees that it is the end */
    for(idx_t d=0; d < data->dims[m]; ++d) {
      idx_t startid = get_next_tileid(TILE_BEGIN, data->dims, nmodes, m, d);

      /* compute start id manually */
      data->coords[m] = d;
      idx_t const manual = get_tile_id(data->dims, nmodes, data->coords);
      ASSERT_EQUAL(manual, startid);

      /* Iterate over all tiles and also check last+1. Start from one because
       * TILE_BEGIN has already happened. */
      idx_t id = startid;
      for(idx_t t=1; t < ntiles; ++t) {
        id = get_next_tileid(id, data->dims, nmodes, m, d);
        ASSERT_NOT_EQUAL(startid, id);
        ASSERT_NOT_EQUAL(TILE_END, id);
      }
      id = get_next_tileid(id, data->dims, nmodes, m, d);
      ASSERT_EQUAL(TILE_END, id);
    }
  }
}


/*
 * Do a traversal of the space with non-uniform tile dimensions.
 */
CTEST2(tile_traverse, tile_weird_dim)
{
  idx_t const nmodes = 5;
  for(idx_t m=0; m < nmodes; ++m) {
    data->dims[m] = m+1;
  }

  for(idx_t m=0; m < nmodes; ++m) {
    /* empty tiles */
    __fill_arr(data->coords, nmodes, 0);

    /* the number of tiles that the traversal should go through */
    idx_t ntiles = 1;
    for(idx_t m2=0; m2 < nmodes; ++m2) {
      if(m2 != m) {
        ntiles *= data->dims[m2];
      }
    }

    /* now ensure every idx in the mode sees that it is the end */
    for(idx_t d=0; d < data->dims[m]; ++d) {
      idx_t startid = get_next_tileid(TILE_BEGIN, data->dims, nmodes, m, d);

      /* compute start id manually */
      data->coords[m] = d;
      idx_t const manual = get_tile_id(data->dims, nmodes, data->coords);
      ASSERT_EQUAL(manual, startid);

      /* Iterate over all tiles and also check last+1. Start from one because
       * TILE_BEGIN has already happened. */
      idx_t id = startid;
      for(idx_t t=1; t < ntiles; ++t) {
        id = get_next_tileid(id, data->dims, nmodes, m, d);
        ASSERT_NOT_EQUAL(startid, id);
        ASSERT_NOT_EQUAL(TILE_END, id);
      }
      id = get_next_tileid(id, data->dims, nmodes, m, d);
      ASSERT_EQUAL(TILE_END, id);
    }
  }
}


/*
 * Test out-of-bounds tile traversals.
 */
CTEST2(tile_traverse, tile_oob)
{
  idx_t const nmodes = 5;
  for(idx_t m=0; m < nmodes; ++m) {
    data->dims[m] = m+1;
  }

  idx_t id;
  id = get_next_tileid(TILE_END, data->dims, nmodes, 0, 0);
  ASSERT_EQUAL(TILE_ERR, id);

  for(idx_t m=0; m < nmodes; ++m) {
    data->dims[m] = m+1;
  }

  for(idx_t m=0; m < nmodes; ++m) {
    /* now ensure every idx in the mode sees that it is the end */
    for(idx_t d=0; d < data->dims[m]; ++d) {
      idx_t id = get_next_tileid(TILE_BEGIN, data->dims, nmodes, m, d);
      while(id != TILE_END) {
        id = get_next_tileid(id, data->dims, nmodes, m, d);
      }
      id = get_next_tileid(id, data->dims, nmodes, m, d);
      ASSERT_EQUAL(TILE_ERR, id);
      id = get_next_tileid(id, data->dims, nmodes, m, d);
      ASSERT_EQUAL(TILE_ERR, id);
    }
  }
}

