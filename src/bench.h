#ifndef SPLATT_BENCH_H
#define SPLATT_BENCH_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "sptensor.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void bench_splatt(
  sptensor_t * const tt,
  matrix_t ** mats,
  idx_t const niters,
  idx_t const nthreads,
  int const scale);

void bench_giga(
  sptensor_t * const tt,
  matrix_t ** mats,
  idx_t const niters,
  idx_t const nthreads,
  int const scale);

void bench_ttbox(
  sptensor_t * const tt,
  matrix_t ** mats,
  idx_t const niters,
  idx_t const nthreads,
  int const scale);

#endif
