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
  int write; /** do we write output to file? */

  /* CPD options */
  idx_t rank;
  idx_t niters;
  double tol; /** cease iterations with improvement drops below tol */


  /* dimension of the distribution (used in MPI) */
  idx_t distribution;
  int mpi_dims[MAX_NMODES];

  /* optimizations */
  idx_t nthreads;
  int tile;
} cpd_opts;


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "ftensor.h"
#include "matrix.h"
#include "splatt_mpi.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
#define cpd_als splatt_cpd_als
void cpd_als(
  ftensor_t ** ft,
  matrix_t ** mats,
  matrix_t ** globmats,
  val_t * const lambda,
  rank_info * const rinfo,
  cpd_opts const * const opts);


#define default_cpd_opts splatt_default_cpd_opts

/**
* @brief Fill a cpd_opts struct with default values.
*
* @param args The cpd_opts struct to fill.
*/
void default_cpd_opts(
  cpd_opts * args);


#endif
