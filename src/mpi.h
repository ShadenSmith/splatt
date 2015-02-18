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
* @brief A structure for MPI rank structures (communicators, etc.).
*/
typedef struct
{
  int rank;
  int npes;
  int np13; /* cube root of npes */
  MPI_Status status;

  idx_t global_nnz;
  idx_t global_dims[MAX_NMODES];

  idx_t mat_start[MAX_NMODES];
  idx_t mat_end[MAX_NMODES];

  /* start/end idxs for each process */
  idx_t * mat_starts[MAX_NMODES];
  idx_t * mat_ends[MAX_NMODES];

  idx_t layer_starts[MAX_NMODES];
  idx_t layer_ends[MAX_NMODES];

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


/**
* @brief Compute a distribution of factor matrices that minimizes communication
*        volume.
*
* @param rinfo MPI structure containing rank and communicator information.
* @param tt A partition of the tensor. NOTE: indices will be reordered after
*           distribution to ensure contiguous matrix partitions.
*/
void mpi_distribute_mats(
  rank_info * const rinfo,
  sptensor_t * const tt);

void mpi_setup_comms(
  rank_info * const rinfo);

void mpi_send_recv_stats(
  rank_info const * const rinfo,
  sptensor_t const * const tt);

#endif
