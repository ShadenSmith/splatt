#ifndef SPLATT_CPD_H
#define SPLATT_CPD_H

#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  /* tensor file */
  char * ifname;

  /* CPD options */
  idx_t niters;
  idx_t rank;

  /* optimizations */
  idx_t nthreads;
  int tile;
} cpd_opts;


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "matrix.h"
#include "splatt_mpi.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void cpd(
  sptensor_t * const tt,
  matrix_t ** mats,
  rank_info * const rinfo,
  cpd_opts const * const opts);


#endif
