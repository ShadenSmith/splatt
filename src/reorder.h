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
  idx_t * perms[MAX_NMODES];  /** Permutation array for each mode */
  idx_t * iperms[MAX_NMODES]; /** Inverse permutation array for each mode */
} permutation_t;



/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "sptensor.h"
#include "ftensor.h"


/******************************************************************************
 * TENSOR REORDER FUNCTIONS
 *****************************************************************************/

#define tt_perm splatt_tt_perm
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


#define build_pptr splatt_build_pptr
/**
* @brief Build a data structure containing the size of each partition (in
*        vertices) and a list of vertices in each partition.
*
* @param parts An array marking which partition each vertex is in.
* @param nparts The number of partitions.
* @param nvtxs The number of vertices.
* @param ret_pptr RETURN: An array marking the size (in vertices) of each
*                         partition.
* @param ret_plookup RETURN: An array indexed by with a list of the vertices
*                            in each partition.
*/
void build_pptr(
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const nvtxs,
  idx_t ** ret_pptr,
  idx_t ** ret_plookup);


#define perm_apply splatt_perm_apply
/**
* @brief Reorders a coordinate tensor based on a permutation for each mode.
*
* @param tt The tensor to reorder.
* @param perm A list of permutations (one for each mode).
*/
void perm_apply(
  sptensor_t * const tt,
  idx_t ** perm);


#define perm_rand splatt_perm_rand
/**
* @brief Computes a random permutation of a tensor, applies the permutation,
*        and returns the permutation.
*
* @param tt The tensor to reorder.
*
* @return The random permutation.
*/
permutation_t * perm_rand(
  sptensor_t * const tt);



#define shuffle_idx splatt_shuffle_idx
/**
* @brief Randomly shuffle a random list of idx_t.
*
* @param arr The array to shuffle.
* @param N The length or 'arr'.
*/
void shuffle_idx(
    idx_t * const arr,
    idx_t const N);


#define perm_hgraph splatt_perm_hgraph
permutation_t * perm_hgraph(
  sptensor_t * const tt,
  ftensor_t const * const ft,
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const mode);


#define perm_graph splatt_perm_graph
permutation_t * perm_graph(
  sptensor_t * const tt,
  idx_t const * const parts,
  idx_t const nparts);


#define perm_identity splatt_perm_identity
permutation_t * perm_identity(
  idx_t const * const dims,
  idx_t const nmodes);


#define perm_alloc splatt_perm_alloc
permutation_t * perm_alloc(
  idx_t const * const dims,
  idx_t const nmodes);


#define perm_free splatt_perm_free
void perm_free(
  permutation_t * perm);


/******************************************************************************
 * MATRIX REORDER FUNCTIONS
 *****************************************************************************/
#define perm_matrix splatt_perm_matrix
matrix_t * perm_matrix(
  matrix_t const * const mat,
  idx_t const * const perm,
  matrix_t * retmat);


#endif
