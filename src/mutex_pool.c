

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
  pool->locks = splatt_malloc(num_locks * pad_size * sizeof(omp_lock_t));
  for(int l=0; l < num_locks; ++l) {
    omp_init_lock(pool->locks + mutex_translate_id(l, num_locks, pad_size));
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

  splatt_free(pool);
}

