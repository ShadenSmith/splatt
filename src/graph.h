#ifndef SPLATT_GRAPH_H
#define SPLATT_GRAPH_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "ftensor.h"


/* GRAPH TYPES */

#ifdef SPLATT_USE_MTMETIS

#include <mtmetis.h>
typedef mtmetis_vtx_t vtx_t;
typedef mtmetis_adj_t adj_t;
typedef mtmetis_wgt_t wgt_t;
typedef mtmetis_pid_t part_t;

#else

typedef splatt_idx_t vtx_t;
typedef splatt_idx_t adj_t;
typedef splatt_idx_t wgt_t;
typedef splatt_idx_t part_t;

#endif

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


/**
* @brief A structure representing a traditional graph.
*/
typedef struct
{
  vtx_t nvtxs;  /* Number of vertices in the graph. */
  adj_t nedges; /* Number of edges in the graph. */

  adj_t * eptr; /** Adjacency list pointer. */
  adj_t * eind; /** Adjacency list. */

  adj_t nvwgts;
  wgt_t * vwgts; /** Vertex weights. */
  wgt_t * ewgts; /** Edge weights. */
} splatt_graph;


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
  splatt_coo const * const tt);


#define hgraph_free splatt_hgraph_free
/**
* @brief Free all memory allocated for a hypergraph. NOTE: this frees the
*        pointer too!
*
* @param hg The hypergraph to free.
*/
void hgraph_free(
  hgraph_t * hg);


#define graph_convert splatt_graph_convert
/**
* @brief Convert an m-way sparse tensor to an m-partite graph. Edges are
*        weighted based on the nnz that connect two indices.
*
* @param tt The tensor to convert.
*
* @return  The m-partite graph.
*/
splatt_graph * graph_convert(
    splatt_coo * const tt);


#define graph_alloc splatt_graph_alloc
/**
* @brief Allocate space for a graph.
*
* @param nvtxs The number of vertices in the graph.
* @param nedges The number of edges. This will be the size of 'eind', so double
*               the number if you want unweighted.
* @param use_vtx_wgts If vertices are weighted, supply non-zero value here.
* @param use_edge_wgts If edges are weighted, supply non-zero value here.
*
* @return An allocated graph structure.
*/
splatt_graph * graph_alloc(
    vtx_t nvtxs,
    adj_t nedges,
    int use_vtx_wgts,
    int use_edge_wgts);


#define graph_free splatt_graph_free
/**
* @brief Free the memory allocated from graph_alloc().
*
* @param graph The graph to free.
*/
void graph_free(
    splatt_graph * graph);


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



#ifdef SPLATT_USE_METIS
#define metis_part splatt_metis_part
/**
* @brief Partition a graph using Metis with default options.
*
* @param graph The graph to partition.
* @param num_partitions The number of partitions to use.
* @param[out] edgecut The edgecut of the resulting partitioning.
*
* @return A partitioning of the vertices.
*/
splatt_idx_t *  metis_part(
    splatt_graph * graph,
    splatt_idx_t const num_partitions,
    splatt_idx_t * const edgecut);
#endif


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
