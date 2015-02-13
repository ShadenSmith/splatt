#ifndef SPLATT_MPI_H
#define SPLATT_MPI_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"
#include <mpi.h>


/******************************************************************************
 * PUBLIC STRUCTURES
 *****************************************************************************/

/**
* @brief A structure for MPI rank structurse (communicators, etc.).
*/
typedef struct
{
  int rank;
  int npes;
  int np13; /* cube root of npes */
  MPI_Status status;

  MPI_Comm comm_3d;
  int rank_3d;
  int dims_3d[MAX_NMODES];
  int coords_3d[MAX_NMODES];
} rank_info;


/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

sptensor_t * mpi_tt_read(
  char const * const ifname,
  rank_info * const rinfo);

void mpi_setup_comms(
  rank_info * const rinfo);

void tt_distribute_stats(
  sptensor_t * const tt,
  idx_t const nprocs);

#endif
