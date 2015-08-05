#ifndef SPLATT_GRAPH_H
#define SPLATT_GRAPH_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "ftensor.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/


/**
* @brief Different routines for determining vertex weights.
*/
typedef enum
{
  VTX_WT_NONE,      /** Unweighted vertices. */
  VTX_WT_FIB_NNZ    /** Weighted based on nnz in the fiber */
} hgraph_vwt_type;


/**
* @brief A structure representing a hypergraph.
*/
typedef struct
{
  idx_t nvtxs;    /** Number of vertices in the hypergraph. */
  idx_t nhedges;  /** Number of hyperedges in the hypergraph. */
  idx_t * vwts;   /** Array of vertex weights. NULL if unweighted. */
  idx_t * hewts;  /** Array of hyperedge weights. NULL if unweighted. */
  idx_t * eptr;   /** Array of length (nhedges+1) and marks start of hedges.
                      Indexes into 'eind'. */
  idx_t * eind;   /** Array containing all vertices that appear in hedges. */
} hgraph_t;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
#define hgraph_fib_alloc splatt_hgraph_fib_alloc
/**
* @brief Allocate and fill a hypergraph from a CSF tensor. The tensor fibers
*        become vertices and modes are mapped to nets. This is a sort of
*        fine-grained model for fibers.
*
* @param ft The CSF tensor to convert.
* @param mode Which mode we are operating on.
*
* @return The hypergraph.
*/
hgraph_t * hgraph_fib_alloc(
  ftensor_t const * const ft,
  idx_t const mode);


#define hgraph_nnz_alloc splatt_hgraph_nnz_alloc
/**
* @brief Allocate and fill a hypergraph from a coordinate tensor. The nonzeros
*        become vertices and modes are mapped to nets. This is a fine-grained
*        model for nonzeros.
*
* @param tt The coordinate tensor to convert.
*
* @return The hypergraph.
*/
hgraph_t * hgraph_nnz_alloc(
  sptensor_t const * const tt);


#define hgraph_free splatt_hgraph_free
/**
* @brief Free all memory allocated for a hypergraph. NOTE: this frees the
*        pointer too!
*
* @param hg The hypergraph to free.
*/
void hgraph_free(
  hgraph_t * hg);


#define hgraph_uncut splatt_hgraph_uncut
/**
* @brief Given a hypergraph partitioning, return a list of the uncut nets.
*
* @param hg The hypergraph to inspect.
* @param parts A partitioning of the vertices.
* @param nnotcut [OUT] This will be set to the number of uncut nets.
*
* @return A list of the 'ncut' uncut nets.
*/
idx_t * hgraph_uncut(
  hgraph_t const * const hg,
  idx_t const * const parts,
  idx_t * const nnotcut);


#ifdef SPLATT_USE_PATOH
#define patoh_part splatt_patoh_part
/**
* @brief Partition a hypergraph using PaToH, optimizing for cut with 'speed'
*        emphasis.
*
* @param hg The hypergraph to partition.
* @param nparts The number of partitions.
*
* @return A partitioning of the hypergraph vertices.
*/
idx_t * patoh_part(
    hgraph_t const * const hg,
    idx_t const nparts);
#endif /* patoh functions */


#ifdef SPLATT_USE_ASHADO
#define ashado_part splatt_ashado_part
/**
* @brief Partition a hypergraph using Ashado.
*
* @param hg The hypergraph to partition.
* @param nparts The number of partitions.
*
* @return A partitioning of the hypergraph vertices.
*/
idx_t * ashado_part(
    hgraph_t const * const hg,
    idx_t const nparts);
#endif /* patoh functions */

#endif
