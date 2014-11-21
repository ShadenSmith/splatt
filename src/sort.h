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
void tt_sort(
  sptensor_t * const tt,
  idx_t const mode,
  idx_t * dim_perm);

void insertion_sort(
  idx_t * const a,
  idx_t const n);

void quicksort(
  idx_t * const a,
  idx_t const n);

#endif
