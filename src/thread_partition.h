/**
* @file thread_partition.h
* @brief Routines for partitioning data among threads. Formerly `ccp/ccp.h`.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2017-11-24
*/

#ifndef SPLATT_THREAD_PARTITION_H
#define SPLATT_THREAD_PARTITION_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define partition_weighted splatt_partition_weighted
/**
* @brief Partition weighted items among threads in a load-balanced manner. This
*        uses chains-on-chains partitioning to optimally load balance (optimal
*        due to integer weights).
*
* @param weights The weights of each item. NOTE: these are destroyed.
* @param nitems The number of items to partition.
* @param nparts The number of partitions.
* @param[out] bottleneck The weight of the heaviest partition.
*
* @return A balanced partitioning of the data. Thread 't' should process items
*         [t, t+1). This should be freed with `splatt_free()`.
*/
idx_t * partition_weighted(
    idx_t * const weights,
    idx_t const nitems,
    idx_t const nparts,
    idx_t * const bottleneck);


#define partition_simple splatt_partition_simple
/**
* @brief Partition unweighted items equally.
*
* @param nitems The number of items.
* @param nparts The number of partitions.
*
* @return A partitioning of the data. Thread 't' should process items [t, t+1).
*         This should be freed with `splatt_free()`.
*/
idx_t * partition_simple(
    idx_t const nitems,
    idx_t const nparts);


#define prefix_sum_inc splatt_prefix_sum_inc
/**
* @brief Compute an inclusive prefix sum: [3, 4, 5] -> [3, 7, 12].
*
* @param weights The numbers to sum.
* @param nitems The number of items in 'weights'.
*/
void prefix_sum_inc(
    idx_t * const weights,
    idx_t const nitems);


#define prefix_sum_exc splatt_prefix_sum_exc
/**
* @brief Compute an exclusive prefix sum: [3, 4, 5] -> [0, 3, 7].
*
* @param weights The numbers to sum.
* @param nitems The number of items in 'weights'.
*/
void prefix_sum_exc(
    idx_t * const weights,
    idx_t const nitems);

#endif
