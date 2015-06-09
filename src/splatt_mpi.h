#ifndef SPLATT_MPI_H
#define SPLATT_MPI_H

/******************************************************************************
 * MPI DISABLED
 *****************************************************************************/
# ifndef SPLATT_USE_MPI
/* Just a dummy for when MPI is not enabled. */
typedef struct
{
  int rank;
} rank_info;
# else



/******************************************************************************
 * FULL MPI SUPPORT
 *****************************************************************************/

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
  idx_t nmodes;
  idx_t global_nnz;
  idx_t global_dims[MAX_NMODES];
  idx_t mat_start[MAX_NMODES];
  idx_t mat_end[MAX_NMODES];
  idx_t layer_starts[MAX_NMODES];
  idx_t layer_ends[MAX_NMODES];

  /* tt indices [ownstart, ownend) are mine. These operate in global indexing,
   * so indmap is used if present. */
  idx_t nowned[MAX_NMODES];      /** number of rows owned */
  idx_t ownstart[MAX_NMODES];
  idx_t ownend[MAX_NMODES];


  /* start/end idxs for each process */
  idx_t * mat_ptrs[MAX_NMODES];

  /* same as cpd_args distribution. */
  idx_t distribution;

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
  int rank_3d;
  int mode_rank[MAX_NMODES];
  int dims_3d[MAX_NMODES];
  int coords_3d[MAX_NMODES];
  int layer_rank[MAX_NMODES];
  int layer_size[MAX_NMODES];

  /* Miscellaneous */
  MPI_Status status;
  MPI_Request req;
  idx_t worksize;
} rank_info;



/**
* @brief Communication pattern type. We support point-to-point, all-to-all
*        (vectorized), and our own sparse reduction pattern (soon).
*/
typedef enum
{
  SPLATT_POINT2POINT,
  SPLATT_ALL2ALL,
  SPLATT_SPARSEREDUCE
} splatt_comm_type;

#define DEFAULT_COMM SPLATT_POINT2POINT


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "reorder.h"
#include "cpd.h"



/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

#define mpi_update_rows splatt_mpi_update_rows
/**
* @brief Do an all-to-all communication of exchanging updated rows with other
*        ranks. We send globmats[mode] to the needing ranks and receive other
*        ranks' globmats entries which we store in mats[mode].
*
* @param indmap The local->global mapping of the tensor. May be NULL if the
*               mapping is identity.
* @param nbr2globs_buf Buffer at least as large as as there are rows to send
*                      (for each rank).
* @param nbr2local_buf Buffer at least as large as there are rows to receive.
* @param localmat Local factor matrix which receives updated values.
* @param globalmat Global factor matrix (owned by me) which is sent to ranks.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the factor matrices.
* @param mode The mode to exchange along.
* @param which Which communication pattern to use.
*/
void mpi_update_rows(
  idx_t const * const indmap,
  val_t * const restrict nbr2globs_buf,
  val_t * const restrict nbr2local_buf,
  matrix_t * const localmat,
  matrix_t * const globalmat,
  rank_info * const rinfo,
  idx_t const nfactors,
  idx_t const mode,
  splatt_comm_type const which);


#define mpi_reduce_rows splatt_mpi_reduce_rows
/**
* @brief Do a reduction (sum) of all neighbor partial products which I own.
*        Updates are written to globalmat.
*
* @param local2nbr_buf A buffer at least as large as nlocal2nbr.
* @param nbr2globs_buf A buffer at least as large as nnbr2globs.
* @param localmat My local matrix containing partial products for other ranks.
* @param globalmat The global factor matrix to update.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the matrices.
* @param mode The mode to operate on.
* @param which Which communication pattern to use.
*/
void mpi_reduce_rows(
  val_t * const restrict local2nbr_buf,
  val_t * const restrict nbr2globs_buf,
  matrix_t const * const localmat,
  matrix_t * const globalmat,
  rank_info * const rinfo,
  idx_t const nfactors,
  idx_t const mode,
  splatt_comm_type const which);


#define mpi_add_my_partials splatt_mpi_add_my_partials
/**
* @brief Add my own partial products to the global matrix that I own.
*
* @param indmap The local->global mapping of the tensor. May be NULL if the
*               mapping is identity.
* @param localmat The local matrix containing my partial products.
* @param globmat The global factor matrix I am writing to.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the matrices.
* @param mode The mode I am operating on.
*/
void mpi_add_my_partials(
  idx_t const * const indmap,
  matrix_t const * const localmat,
  matrix_t * const globmat,
  rank_info const * const rinfo,
  idx_t const nfactors,
  idx_t const mode);


#define mpi_write_mats splatt_mpi_write_mats
/**
* @brief Write distributed matrices to 'basename<N>.mat'.
*
* @param mats The distributed matrices to write to disk.
* @param perm Any row permutation that we must undo.
* @param rinfo MPI rank information.
* @param basename Matrices are written to file 'basename'N.mat.
* @param nmodes The number of matrices to write.
*/
void mpi_write_mats(
  matrix_t ** mats,
  permutation_t const * const perm,
  rank_info const * const rinfo,
  char const * const basename,
  idx_t const nmodes);


#define mpi_write_part splatt_mpi_write_part
/**
* @brief Write a tensor to file <rank>.part. All local indices are converted to
*        global.
*
* @param tt The tensor to write.
* @param perm Any permutations that have been done on the tensor
*             (before compression).
* @param rinfo MPI rank information.
*/
void mpi_write_part(
  sptensor_t const * const tt,
  permutation_t const * const perm,
  rank_info const * const rinfo);


#define mpi_compute_ineed splatt_mpi_compute_ineed
/**
* @brief
*
* @param rinfo
* @param tt
*/
void mpi_compute_ineed(
  rank_info * const rinfo,
  sptensor_t const * const tt,
  idx_t const mode,
  idx_t const nfactors,
  idx_t const distribution);


#define mpi_tt_read splatt_mpi_tt_read
/**
* @brief Each rank reads their 3D partition of a tensor.
*
* @param ifname The file containing the tensor.
* @param rinfo Rank information.
* 
* @return The rank's subtensor.
*/
sptensor_t * mpi_tt_read(
  char const * const ifname,
  rank_info * const rinfo);


#define mpi_filter_tt_1d splatt_mpi_filter_tt_1d
/**
* @brief Run nonzeros from tt through filter to 'ftt'. This is 1D filtering,
*        so we accept any nonzeros whose ind[mode] are within [start, end).
*
* @param mode The mode to filter along.
* @param tt The original tensor.
* @param ftt The tensor to filter into (pre-allocated).
* @param start The first index to accept (inclusive).
* @param end The last index to accept (exclusive).
*/
void mpi_filter_tt_1d(
  idx_t const mode,
  sptensor_t const * const tt,
  sptensor_t * const ftt,
  idx_t start,
  idx_t end);


#define mpi_distribute_mats splatt_mpi_distribute_mats
/**
* @brief Compute a distribution of factor matrices that minimizes communication
*        volume.
*
* @param rinfo MPI structure containing rank and communicator information.
* @param tt A partition of the tensor. NOTE: indices will be reordered after
*           distribution to ensure contiguous matrix partitions.
* @param distribution The dimension of the distribution to perform (1-3).
*
* @return The permutation that was applied to tt.
*/
permutation_t *  mpi_distribute_mats(
  rank_info * const rinfo,
  sptensor_t * const tt,
  idx_t const distribution);


#define mpi_find_owned splatt_mpi_find_owned
/**
* @brief Setup 'owned' structures which mark the location of owned rows in
*        my local tensor.
*
* @param tt My subtensor.
* @param rinfo MPI rank information.
*/
void mpi_find_owned(
  sptensor_t const * const tt,
  idx_t const mode,
  rank_info * const rinfo);


#define mpi_setup_comms splatt_mpi_setup_comms
/**
* @brief Fill rinfo with process' MPI rank information. Includes rank, 3D
*        communicator, etc.
*
* @param rinfo The rank data structure.
*/
void mpi_setup_comms(
  rank_info * const rinfo);


#define rank_free splatt_rank_free
/**
* @brief Free structures allocated inside rank_info.
*
* @param rinfo The rank structure to free.
* @param nmodes The number of modes that have been allocated.
*/
void rank_free(
  rank_info rinfo,
  idx_t const nmodes);


#define mpi_time_stats splatt_mpi_time_stats
/**
* @brief Update timers[] with max values on the master rank instead of only
*        local times.
*
* @param rinfo Struct containing rank information.
*/
void mpi_time_stats(
  rank_info const * const rinfo);


#define mpi_send_recv_stats splatt_mpi_send_recv_stats
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

#endif /* SPLATT_USE_MPI */
#endif
