#ifndef SPLATT_CCP_CCP_H
#define SPLATT_CCP_CCP_H

#include "../base.h"


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <stdbool.h>




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define partition_1d splatt_partition_1d
/**
* @brief Compute the optimal 1D partitioning of 'weights'.
*
* @param weights An array of workload weights, length 'nitems'.
* @param nitems The number of items we are partitioning.
* @param[out] parts A ptr into weights, marking each partition. THIS IS ASSUMED
*                   to be pre-allocated at least of size 'nparts+1'.
* @param nparts The number of partitions to compute.
*
* @return The amount of work in the largest partition (i.e., the bottleneck).
*/
idx_t partition_1d(
    idx_t * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts);


#define lprobe splatt_lprobe
/**
* @brief Attempt to partition 'weights' with each process having at most
*        'bottleneck' amount of work.
*
* @param weights An array of workload weights, length 'nitems'.
* @param nitems The number of items we are partitioning.
* @param[out] parts A ptr into weights, marking each partition. THIS IS ASSUMED
*                   to be pre-allocated at least of size 'nparts+1'.
* @param nparts The number of partitions to compute.
* @param bottleneck The maximum partition size.
*
* @return Returns true if it was able to successfully partition, false
*         otherwise.
*/
bool lprobe(
    idx_t const * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts,
    idx_t const bottleneck);


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
