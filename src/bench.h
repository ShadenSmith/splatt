#ifndef SPLATT_BENCH_H
#define SPLATT_BENCH_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "sptensor.h"
#include "reorder.h"



/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  idx_t niters;
  idx_t * threads;
  idx_t nruns;
  int write;
  int tile;
  permutation_t * perm;
} bench_opts;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void bench_splatt(
  sptensor_t * const tt,
  matrix_t ** mats,
  bench_opts const * const opts);

void bench_giga(
  sptensor_t * const tt,
  matrix_t ** mats,
  bench_opts const * const opts);

void bench_ttbox(
  sptensor_t * const tt,
  matrix_t ** mats,
  bench_opts const * const opts);


#endif
