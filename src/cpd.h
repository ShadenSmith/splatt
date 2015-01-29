#ifndef SPLATT_CPD_H
#define SPLATT_CPD_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "sptensor.h"


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
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void cpd(
  sptensor_t * const tt,
  matrix_t ** mats,
  cpd_opts const * const opts);


#endif
