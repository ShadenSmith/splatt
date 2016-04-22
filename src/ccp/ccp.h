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
idx_t partition_1d(
    idx_t * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts);


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
