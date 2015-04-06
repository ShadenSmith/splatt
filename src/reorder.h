#ifndef SPLATT_REORDER_H
#define SPLATT_REORDER_H

#include "base.h"



/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

/**
* @brief The type of reordering to perform.
*/
typedef enum
{
  PERM_RAND,
  PERM_GRAPH,       /** Reordering based on an n-partite graph partitioning. */
  PERM_HGRAPH,      /** Reordering based on an hypergraph partitioning. */
  PERM_FIBSCHED,    /** Not done. */
  PERM_ERROR,
} splatt_perm_type;


/**
* @brief Structure for containing the permutation and inverse permutations of
*        a tensor.
*/
typedef struct
{
  idx_t * perms[MAX_NMODES];
  idx_t * iperms[MAX_NMODES];
} permutation_t;



/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "sptensor.h"
#include "ftensor.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Permute a tensor.
*
* @param tt The tensor to permute.
* @param type The type of permutation to perform.
* @param mode The mode to operate on if applicable.
* @param pfile The partitioning file, if applicable.
*
* @return
*/
permutation_t *  tt_perm(
  sptensor_t * const tt,
  splatt_perm_type const type,
  idx_t const mode,
  char const * const pfile);


void build_pptr(
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const nvtxs,
  idx_t ** ret_pptr,
  idx_t ** ret_plookup);


void perm_apply(
  sptensor_t * const tt,
  idx_t ** perm);


permutation_t * perm_rand(
  sptensor_t * const tt);


permutation_t * perm_hgraph(
  sptensor_t * const tt,
  ftensor_t const * const ft,
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const mode);


permutation_t * perm_graph(
  sptensor_t * const tt,
  idx_t const * const parts,
  idx_t const nparts);

permutation_t * perm_identity(
  idx_t const * const dims,
  idx_t const nmodes);

permutation_t * perm_alloc(
  idx_t const * const dims,
  idx_t const nmodes);


void perm_free(
  permutation_t * perm);


/******************************************************************************
 * MATRIX REORDER FUNCTIONS
 *****************************************************************************/
matrix_t * perm_matrix(
  matrix_t const * const mat,
  idx_t const * const perm,
  matrix_t * retmat);



#endif
