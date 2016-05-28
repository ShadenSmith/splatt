#ifndef SPLATT_STATS_H
#define SPLATT_STATS_H

#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

/**
* @brief The types of tensor statistics available.
*/
typedef enum
{
  STATS_BASIC,    /** Dimensions, nonzero count, and density. */
  STATS_HPARTS,   /** Hypergraph partitioning information. Requires MODE */
  STATS_ERROR,
} splatt_stats_type;



/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "csf.h"
#include "splatt_mpi.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
#define stats_tt splatt_stats_tt
/**
* @brief Output statistics about a sparse tensor.
*
* @param tt The sparse tensor to inspect.
* @param ifname The filename of the tensor. Can be NULL.
* @param type The type of statistics to output.
* @param mode The mode of tt to operate on, if applicable.
* @param pfile The partitioning file to work with, if applicable.
*/
void stats_tt(
  sptensor_t * const tt,
  char const * const ifname,
  splatt_stats_type const type,
  idx_t const mode,
  char const * const pfile);


#define stats_csf splatt_stats_csf
/**
* @brief Output statistics about a CSF tensor.
*
* @param ct The CSF tensor to analyze.
*/
void stats_csf(
  splatt_csf const * const ct);


#define cpd_stats splatt_cpd_stats
/**
* @brief Output work-related statistics before a CPD factorization. This
*        includes rank, #threads, tolerance, etc.
*
* @param csf The CSF tensor we are factoring.
* @param nfactors The number of factors.
* @param opts Other CPD options.
*/
void cpd_stats(
  splatt_csf const * const csf,
  idx_t const nfactors,
  double const * const opts);

void cpd_stats2(
    idx_t const rank,
    idx_t const num_modes,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts);

/******************************************************************************
 * MPI FUNCTIONS
 *****************************************************************************/
#ifdef SPLATT_USE_MPI

#define mpi_cpd_stats splatt_mpi_cpd_stats
/**
* @brief Output work-related statistics before a CPD factorization. This
*        includes rank, #threads, tolerance, etc.
*
* @param csf The CSF tensor we are factoring.
* @param nfactors The number of factors.
* @param opts Other CPD options.
* @param rinfo MPI rank information.
*/
void mpi_cpd_stats(
  splatt_csf const * const csf,
  idx_t const nfactors,
  double const * const opts,
  rank_info * const rinfo);


#define mpi_global_stats splatt_mpi_global_stats
/**
* @brief Copy global information into local tt, print statistics, and
*        restore local information.
*
* @param tt The tensor to hold global information.
* @param rinfo Global tensor information.
*/
void mpi_global_stats(
  sptensor_t * const tt,
  rank_info * const rinfo,
  char const * const ifname);

#define mpi_rank_stats splatt_mpi_rank_stats

/**
* @brief Output statistics about MPI rank information.
*
* @param tt The tensor we are operating on.
* @param rinfo MPI rank information.
* @param args Some CPD parameters.
*/
void mpi_rank_stats(
  sptensor_t const * const tt,
  rank_info const * const rinfo);

#endif /* endif SPLATT_USE_MPI */

#endif
