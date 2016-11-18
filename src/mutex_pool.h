#ifndef SPLATT_MUTEX_POOL_H
#define SPLATT_MUTEX_POOL_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <stdbool.h>

#ifdef _OPENMP
#include <omp.h>
#endif


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/


/**
* @brief A pool of mutexes for synchronization.
*/
typedef struct
{
  bool initialized;
  int num_locks;
  int pad_size;

#ifdef _OPENMP
  omp_lock_t * locks;
#else
  volatile int * locks;
#endif
} mutex_pool; 


#ifndef SPLATT_DEFAULT_NLOCKS
#define SPLATT_DEFAULT_NLOCKS 1024
#endif


#ifndef SPLATT_DEFAULT_LOCK_PAD
#define SPLATT_DEFAULT_LOCK_PAD 16
#endif


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define mutex_alloc splatt_mutex_alloc
/**
* @brief Allocate a pool of mutexes following SPLATT default values. Defaults
*        are chosen to provide high MTTKRP performance.
*
* @return The allocated mutex pool.
*/
mutex_pool * mutex_alloc();


#define mutex_alloc_custom splatt_mutex_alloc_custom
/**
* @brief Allocate a pool of mutexes with custom specifications.
*
* @param num_locks The number of mutexes to use.
* @param pad_size The padding between mutexes.
*
* @return  The allocated mutex pool.
*/
mutex_pool * mutex_alloc_custom(
    int const num_locks,
    int const pad_size);


#define mutex_free splatt_mutex_free
/**
* @brief Free the memory allocated for a mutex pool.
*
* @param pool The pool to free.
*/
void mutex_free(
    mutex_pool * pool);


#define mutex_translate_id splatt_mutex_translate_id
/**
* @brief Convert an arbitrary integer ID to a lock ID in a mutex pool.
*
* @param id An arbitrary integer ID (e.g., matrix row).
* @param num_locks The size of the mutex pool.
* @param pad_size The padding between each lock.
*
* @return 
*/
static inline int mutex_translate_id(
    int const id,
    int const num_locks,
    int const pad_size)
{
  return (id % num_locks) * pad_size;
}


#define mutex_set_lock splatt_mutex_set_lock
/**
* @brief Claim a lock of a mutex pool. The lock is identified with an ID,
*        which is an arbitrary integer which uniquely identifies the memory to
*        protect (e.g., a matrix row). The ID is then translated into an actual
*        lock based on the pool size and padding.
*
* @param pool The pool to use.
* @param id The ID of the lock.
*/
static inline void mutex_set_lock(
    mutex_pool * const pool,
    int const id)
{
#ifdef _OPENMP
  int const lock_id = mutex_translate_id(id, pool->num_locks, pool->pad_size);
  omp_set_lock(pool->locks + lock_id);
#endif
}



#define mutex_unset_lock splatt_mutex_unset_lock
/**
* @brief Release a lock set with splatt_set_lock().
*
* @param pool The pool containing the lock.
* @param id The ID of the lock, not considering padding or pool size.
*/
static inline void mutex_unset_lock(
    mutex_pool * const pool,
    int const id)
{
#ifdef _OPENMP
  int const lock_id = mutex_translate_id(id, pool->num_locks, pool->pad_size);
  omp_unset_lock(pool->locks + lock_id);
#endif
}


#endif
