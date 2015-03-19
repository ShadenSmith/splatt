#ifndef SPLATT_MPI_H
#define SPLATT_MPI_H


# ifndef USE_MPI
/* Just a dummy void for when MPI is not enabled. */
typedef void * rank_info;

/* FULL MPI SUPPORT */
# else

#include "base.h"
#include <mpi.h>


/******************************************************************************
 * PUBLIC STRUCTURES
 *****************************************************************************/

/**
* @brief A structure for MPI rank structures (communicators, etc.).
*/
typedef struct
{
  /* Tensor information */
  idx_t global_nnz;
  idx_t global_dims[MAX_NMODES];
  idx_t mat_start[MAX_NMODES];
  idx_t mat_end[MAX_NMODES];
  idx_t layer_starts[MAX_NMODES];
  idx_t layer_ends[MAX_NMODES];

  idx_t nowned[MAX_NMODES];      /** number of rows owned */
  idx_t ownstart[MAX_NMODES];    /** tt indices [ownstart, ownend) are mine */
  idx_t ownend[MAX_NMODES];


  /* start/end idxs for each process */
  idx_t * mat_ptrs[MAX_NMODES];

  /* Send/Recv Structures
   * nlocal2nbr: This is the number of rows that I have in my tensor but do not
   *             own. I must send nlocal2nbr partial products AND receive
   *             nlocal2nbr updated rows after each iteration.
   *
   * nnbr2globs: This is the number of rows that other ranks use but I own,
   *             summed across all ranks. I receive this many partial updates
   *             and send this many updated rows after each iteration.
   *
   * local2nbr: These are rows that I compute for but do not own. These partial
   *            products must be sent to neigbors.
   *
   * nbr2local: These are neigbors' rows that I need for MTTKRP. For every row
   *            in local2nbr I need their updated factor matrices.
   *
   * nbr2globs: These are rows that neigbors have but I own. These partial
   *            products are received and I update global matrices with them.
   */
  idx_t nlocal2nbr[MAX_NMODES];
  idx_t nnbr2globs[MAX_NMODES];
  idx_t * nbr2globs_inds[MAX_NMODES];
  idx_t * local2nbr_inds[MAX_NMODES];
  idx_t * nbr2local_inds[MAX_NMODES];
  int   * local2nbr_ptr[MAX_NMODES];
  int   * nbr2globs_ptr[MAX_NMODES];
  int   * local2nbr_disp[MAX_NMODES];
  int   * nbr2globs_disp[MAX_NMODES];


  /* Communicators */
  MPI_Comm comm_3d;
  MPI_Comm layer_comm[MAX_NMODES];

  /* Rank information */
  int rank;
  int npes;
  int np13; /* cube root of npes */
  int rank_3d;
  int mode_rank[MAX_NMODES];
  int dims_3d[MAX_NMODES];
  int coords_3d[MAX_NMODES];
  int layer_rank[MAX_NMODES];


  /* Miscellaneous */
  MPI_Status status;
  MPI_Request req;
  idx_t worksize;
} rank_info;



/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "reorder.h"
#include "cpd.h"



/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/


/**
* @brief
*
* @param tt
* @param mats
* @param globmats
* @param rinfo
* @param opts
*/
void mpi_cpd(
  sptensor_t * const tt,
  matrix_t ** mats,
  matrix_t ** globmats,
  rank_info * const rinfo,
  cpd_opts const * const opts);


void mpi_write_mats(
  matrix_t ** mats,
  permutation_t const * const perm,
  rank_info const * const rinfo,
  char const * const basename,
  idx_t const nmodes);


/**
* @brief
*
* @param rinfo
* @param tt
*/
void mpi_compute_ineed(
  rank_info * const rinfo,
  sptensor_t const * const tt,
  idx_t const nfactors);

/**
* @brief Each rank reads their 3D partition of a tensor.
*
* @param ifname The file containing the tensor.
* @param rinfo Rank information, assumes mpi_setup_comms() has been called
*              first!
*
* @return The rank's subtensor.
*/
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
*
* @return The permutation that was applied to tt.
*/
permutation_t *  mpi_distribute_mats(
  rank_info * const rinfo,
  sptensor_t * const tt);

/**
* @brief Fill rinfo with process' MPI rank information. Includes rank, 3D
*        communicator, etc.
*
* @param rinfo The rank data structure.
*/
void mpi_setup_comms(
  rank_info * const rinfo);


/**
* @brief Free structures allocated inside rank_info.
*
* @param rinfo The rank structure to free.
* @param nmodes The number of modes that have been allocated.
*/
void rank_free(
  rank_info rinfo,
  idx_t const nmodes);


/**
* @brief Print send/recieve information to STDOUT.
*
* @param rinfo MPI rank information. Assumes mpi_distribute_mats() has already
*              been called.
* @param tt The distributed tensor.
*/
void mpi_send_recv_stats(
  rank_info const * const rinfo,
  sptensor_t const * const tt);

#endif /* USE_MPI */
#endif
