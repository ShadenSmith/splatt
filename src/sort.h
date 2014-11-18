#ifndef SPLATT_SORT_H
#define SPLATT_SORT_H

#include "base.h"
#include "sptensor.h"

void insertion_sort(
  idx_t * const a,
  idx_t const n);

void quicksort(
  idx_t * const a,
  idx_t const n);

void tt_sort(
  sptensor_t * const tt,
  idx_t const mode,
  idx_t * dim_perm);

#endif
