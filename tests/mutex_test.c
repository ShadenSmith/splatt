
#include "ctest/ctest.h"
#include "splatt_test.h"

#include "../src/base.h"
#include "../src/mutex_pool.h"

CTEST_DATA(mutex)
{
  int num_ints;
  int num_incs;
  int * counts;
};

CTEST_SETUP(mutex)
{
  data->num_ints = 10;
  data->num_incs = 128;
  data->counts = splatt_malloc(data->num_ints * sizeof(*data->counts));

  for(int i=0; i < data->num_ints; ++i) {
    data->counts[i] = 0;
  }
}

CTEST_TEARDOWN(mutex)
{
  splatt_free(data->counts);
}


CTEST2(mutex, alloc)
{
  mutex_pool * pool = mutex_alloc();
  ASSERT_NOT_NULL(pool);

  ASSERT_EQUAL(SPLATT_DEFAULT_NLOCKS, pool->num_locks);
  ASSERT_EQUAL(SPLATT_DEFAULT_LOCK_PAD, pool->pad_size);
  ASSERT_NOT_NULL(pool->locks);

  mutex_free(pool);
}



CTEST2(mutex, alloc_custom)
{
  mutex_pool * pool = mutex_alloc_custom(10, 100);
  ASSERT_NOT_NULL(pool);

  ASSERT_EQUAL(10, pool->num_locks);
  ASSERT_EQUAL(100, pool->pad_size);
  ASSERT_NOT_NULL(pool->locks);

  mutex_free(pool);
}


#ifdef _OPENMP
CTEST2(mutex, omp_lock)
{
  mutex_pool * pool = mutex_alloc();

  int num_threads = 4;

  #pragma omp parallel num_threads(num_threads) shared(pool)
  {
    for(int i=0; i < data->num_ints; ++i) {
      for(int x=0; x < data->num_incs; ++x) {
        mutex_set_lock(pool, i);
        ++(data->counts[i]);
        mutex_unset_lock(pool, i);
      }
    }
  } /* end omp parallel */

  for(int i=0; i < data->num_ints; ++i) {
    ASSERT_EQUAL(num_threads * data->num_incs, data->counts[i]);
  }

  mutex_free(pool);
}
#endif

