#ifndef SPLATT_BENCH_H
#define SPLATT_BENCH_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "coo.h"
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
  splatt_coo * const tt,
  matrix_t ** mats,
  bench_opts const * const opts);

void bench_csf(
  splatt_coo * const tt,
  matrix_t ** mats,
  bench_opts const * const opts);

void bench_giga(
  splatt_coo * const tt,
  matrix_t ** mats,
  bench_opts const * const opts);

void bench_ttbox(
  splatt_coo * const tt,
  matrix_t ** mats,
  bench_opts const * const opts);

void bench_coord(
  splatt_coo * const tt,
  matrix_t ** mats,
  bench_opts const * const opts);

#endif
