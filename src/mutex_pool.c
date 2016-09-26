

#include "base.h"
#include "mutex_pool.h"



mutex_pool * mutex_alloc_custom(
    int const num_locks,
    int const pad_size)
{
  mutex_pool * pool = splatt_malloc(sizeof(*pool));

  pool->num_locks = num_locks;
  pool->pad_size = pad_size;
  
#ifdef _OPENMP
  pool->locks = splatt_malloc(num_locks * pad_size * sizeof(*pool->locks));
  for(int l=0; l < num_locks; ++l) {
    int const lock = mutex_translate_id(l, num_locks, pad_size);
    omp_init_lock(pool->locks + lock);
  }

#else
  pool->locks = NULL;
#endif

  return pool;
}


mutex_pool * mutex_alloc()
{
  return mutex_alloc_custom(SPLATT_DEFAULT_NLOCKS, SPLATT_DEFAULT_LOCK_PAD);
}


void mutex_free(
    mutex_pool * pool)
{
#ifdef _OPENMP
  for(int l=0; l < pool->num_locks; ++l) {
    int const lock = mutex_translate_id(l, pool->num_locks, pool->pad_size);
    omp_destroy_lock(pool->locks + lock);
  }
#endif

  splatt_free(pool->locks);
  splatt_free(pool);
}

