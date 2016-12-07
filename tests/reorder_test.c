
#include "ctest/ctest.h"
#include "../src/reorder.h"
#include "../src/sort.h"
#include "splatt_test.h"


CTEST_DATA(reorder)
{
  idx_t N;
  idx_t * fororder;
  idx_t * buffer;
};

CTEST_SETUP(reorder)
{
  data->N = 11111;
  data->fororder = splatt_malloc(data->N * sizeof(*data->fororder));
  data->buffer = splatt_malloc(data->N * sizeof(*data->buffer));

  for(idx_t x=0; x < data->N; ++x) {
    data->fororder[x] = x;
  }
}

CTEST_TEARDOWN(reorder)
{
  splatt_free(data->fororder);
  splatt_free(data->buffer);
}


CTEST2(reorder, shuffle)
{
  srand(1);

  memcpy(data->buffer, data->fororder, data->N * sizeof(*data->buffer));

  /* do this 10 times */
  for(idx_t e=0; e < 10; ++e) {
    shuffle_idx(data->buffer, data->N);
    idx_t same = 0;
    for(idx_t x=0; x < data->N; ++x) {
      if(data->fororder[x] == data->buffer[x]) {
        ++same;
      }
    }

    /* arbitrary */
    if(same > 4) {
      ASSERT_FAIL();
    }

    /* check for duplicate/missing values */
    memcpy(data->fororder, data->buffer, data->N * sizeof(*data->buffer));
    quicksort(data->fororder, data->N);
    for(idx_t x=0; x < data->N; ++x) {
      if(data->fororder[x] != x) {
        ASSERT_FAIL();
      }
    }
  }
}
