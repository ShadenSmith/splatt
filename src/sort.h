#ifndef SPLATT_SORT_H
#define SPLATT_SORT_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define tt_sort splatt_tt_sort
/**
* @brief Sort a tensor using a permutation of its modes. Sorting uses dim_perm
*        to order modes by decreasing priority. If dim_perm = {1, 0, 2} then
*        nonzeros will be ordered by ind[1], with ties broken by ind[0], and
*        finally deferring to ind[2].
*
* @param tt The tensor to sort.
* @param mode The primary for sorting.
* @param dim_perm An permutation array that defines sorting priority. If NULL,
*                 a default ordering of {0, 1, ..., m} is used.
*/
void tt_sort(
  sptensor_t * const tt,
  idx_t const mode,
  idx_t * dim_perm);


#define tt_sort_range splatt_tt_sort_range
/**
* @brief Sort a tensor using tt_sort on only a range of the nonzero elements.
*        Nonzeros in the range [start, end) will be sorted.
*
* @param tt The tensor to sort.
* @param mode The primary for sorting.
* @param dim_perm An permutation array that defines sorting priority. If NULL,
*                 a default ordering of {0, 1, ..., m} is used.
* @param start The first nonzero to include in the sorting.
* @param end The end of the nonzeros to sort (exclusive).
*/
void tt_sort_range(
  sptensor_t * const tt,
  idx_t const mode,
  idx_t * dim_perm,
  idx_t const start,
  idx_t const end);


#define insertion_sort splatt_insertion_sort
/**
* @brief An in-place insertion sort implementation for idx_t's.
*
* @param[out] a The array to sort.
* @param n The number of items to sort.
*/
void insertion_sort(
  idx_t * const a,
  idx_t const n);


#define quicksort splatt_quicksort
/**
* @brief An in-place quicksort implementation for idx_t's.
*
* @param[out] a The array to sort.
* @param n The number of items to sort.
*/
void quicksort(
  idx_t * const a,
  idx_t const n);


#define insertion_sort_perm splatt_insertion_sort_perm
/**
* @brief An in-place insertion sort implementation for idx_t's that tracks the
*        resulting permutation of elements.
*
* @param[out] a The array to sort.
* @param[out] perm A permutation array.
* @param n The number of items to sort.
*/
void insertion_sort_perm(
  idx_t * const restrict a,
  idx_t * const restrict perm,
  idx_t const n);


#define quicksort_perm splatt_quicksort_perm
/**
* @brief An in-place quicksort implementation for idx_t's that tracks the
*        resulting permutation of elements.
*
* @param[out] a The array to sort.
* @param[out] perm A permutation array.
* @param n The number of items to sort.
*/
void quicksort_perm(
  idx_t * const restrict a,
  idx_t * const restrict perm,
  idx_t const n);

#endif
