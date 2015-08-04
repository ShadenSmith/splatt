
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"
#include "ftensor.h"
#include "graph.h"
#include "io.h"
#include "matrix.h"
#include "convert.h"
#include "sort.h"
#include "stats.h"
#include "timer.h"


/******************************************************************************
 * TYPES
 *****************************************************************************/

/**
* @brief Key-value pair.
*/
typedef struct
{
  idx_t v;
  unsigned int cnt;
} kvp_t;

static idx_t nreallocs;
static idx_t const ADJ_START_ALLOC = 8;


/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Search an adjacency list and either increase the edge weight if
*        (u,v) is already found, or add the edge if it is not.
*
* @param u The origin vertex.
* @param v The destination vertex.
* @param adj The adjacency list stored in key-value pairs.
* @param adjmkr Marks into adj for each vertex.
* @param adjsize The length of the adjacency list.
* @param nedges The number of edges in the graph.
*/
static inline void __update_adj(
  idx_t const u,
  idx_t const v,
  kvp_t * * const adj,
  idx_t * const adjmkr,
  idx_t * const adjsize,
  idx_t * nedges)
{
  /* search u's adj for v */
  for(idx_t i=0; i < adjmkr[u]; ++i) {
    if(adj[u][i].v == v) {
      adj[u][i].cnt += 1;
      return;
    }
  }

  /* not found, add vertex to adj list */
  if(adjmkr[u] == adjsize[u]) {
    /* resize if necessary */
    adjsize[u] *= 2;
    adj[u] = (kvp_t *) realloc(adj[u], adjsize[u] * sizeof(kvp_t));
    ++nreallocs;
  }
  adj[u][adjmkr[u]].v   = v;
  adj[u][adjmkr[u]].cnt = 1;
  adjmkr[u] += 1;
  *nedges += 1;
}


/**
* @brief Convert a sparse tensor to a tripartite graph. Each slice becomes a
*        vertex and they are connected by nonzero entries. The graph is written
*        to 'ofname'.
*
* @param tt The sparse tensor to convert.
* @param ofname The filename to write to.
*/
static void __convert_ijk_graph(
  sptensor_t * const tt,
  char const * const ofname)
{
  FILE * fout;
  if(ofname == NULL || strcmp(ofname, "-") == 0) {
    fout = stdout;
  } else {
    fout = fopen(ofname, "w");
  }

  nreallocs = 0;

  idx_t nvtxs = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    nvtxs += tt->dims[m];
  }

  /* allocate adj list */
  kvp_t ** adj = (kvp_t **) malloc(nvtxs * sizeof(kvp_t *));
  idx_t * adjmkr  = (idx_t *) malloc(nvtxs * sizeof(idx_t));
  idx_t * adjsize = (idx_t *) malloc(nvtxs * sizeof(idx_t));
  for(idx_t v=0; v < nvtxs; ++v) {
    adjsize[v] = ADJ_START_ALLOC;
    adj[v] = (kvp_t *) malloc(adjsize[v] * sizeof(kvp_t));
    adjmkr[v] = 0;
  }
  /* marks #edges in each adj list and tells us when to resize */

  /* count edges in graph */
  idx_t nedges = 0;
  for(idx_t n=0; n < tt->nnz; ++n) {
    if(n % 100000 == 0) {
      printf("n: %"SPLATT_PF_IDX"\n", n);
    }
    idx_t uoffset = 0;
    /* update each adj list */
    for(idx_t m=0; m < tt->nmodes; ++m) {
      idx_t const u = tt->ind[m][n] + uoffset;
      idx_t voffset = 0;
      /* emit triangle */
      for(idx_t m2=0; m2 < tt->nmodes; ++m2) {
        if(m != m2) {
          idx_t const v = tt->ind[m2][n] + voffset;
          __update_adj(u, v, adj, adjmkr, adjsize, &nedges);
        }
        voffset += tt->dims[m2];
      }
      uoffset += tt->dims[m];
    }
  }

  nedges /= 2;

  /* print header */
  fprintf(fout, "%"SPLATT_PF_IDX" %"SPLATT_PF_IDX" 001\n", nvtxs, nedges);

  /* now write adj list */
  for(idx_t u=0; u < nvtxs; ++u) {
    for(idx_t v=0; v < adjmkr[u]; ++v) {
      fprintf(fout, "%"SPLATT_PF_IDX" %u ", 1+adj[u][v].v, adj[u][v].cnt);
    }
    fprintf(fout, "\n");
  }

  printf("reallocs: %"SPLATT_PF_IDX"\n", nreallocs);

  /* cleanup */
  if(ofname != NULL || strcmp(ofname, "-") != 0) {
    fclose(fout);
  }

  for(idx_t v=0; v < nvtxs; ++v) {
    free(adj[v]);
  }
  free(adj);
  free(adjmkr);
  free(adjsize);
}


/**
* @brief Convert a sparse tensor to a hypergraph and write to 'ofname'. The
*        hypergraph uses the mode-N fibers as vertices and the sparsity
*        pattern to connect all fibers with their contained vertices (via
*        hyperedges).
*
* @param tt The tensor to convert.
* @param mode The mode to operate on.
* @param ofname The filename to write to.
*/
static void __convert_fib_hgraph(
  sptensor_t * tt,
  idx_t const mode,
  char const * const ofname)
{
  ftensor_t ft;
  ften_alloc(&ft, tt, mode, SPLATT_NOTILE);

  hgraph_t * hg = hgraph_fib_alloc(&ft, mode);
  hgraph_write(hg, ofname);

  ften_free(&ft);
  hgraph_free(hg);
}


/**
* @brief Convert a sparse tensor to a hypergraph and write to 'ofname'. The
*        hypergraph uses the nonzeros as vertices and the sparsity pattern
*        to connect all vertices to <nmodes> hyperedges.
*
* @param tt The tensor to convert.
* @param ofname The filename to write to.
*/
static void __convert_nnz_hgraph(
  sptensor_t const * const tt,
  char const * const ofname)
{
  hgraph_t * hg = hgraph_nnz_alloc(tt);

  hgraph_write(hg, ofname);
  hgraph_free(hg);
}


/**
* @brief Converts a sparse tensor into a CSR matrix whose rows are the
*        mode-N fibers. This is equivalent to the transpose of the mode-N
*        unfolding. The CSR matrix is written to a file.
*
* @param tt The sparse tensor to convert.
* @param mode The mode to operate on.
* @param ofname The filename to write the matrix to.
*/
static void __convert_fib_mat(
  sptensor_t * tt,
  idx_t const mode,
  char const * const ofname)
{
  ftensor_t ft;
  ften_alloc(&ft, tt, mode, 0);
  spmatrix_t * mat = ften_spmat(&ft);

  spmat_write(mat, ofname);

  spmat_free(mat);
  ften_free(&ft);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_convert(
  char const * const ifname,
  char const * const ofname,
  idx_t const mode,
  splatt_convert_type const type)
{
  sptensor_t * tt = tt_read(ifname);
  stats_tt(tt, ifname, STATS_BASIC, 0, NULL);

  timer_start(&timers[TIMER_CONVERT]);

  switch(type) {
  case CNV_IJK_GRAPH:
    __convert_ijk_graph(tt, ofname);
    break;
  case CNV_FIB_HGRAPH:
    __convert_fib_hgraph(tt, mode, ofname);
    break;
  case CNV_NNZ_HGRAPH:
    __convert_nnz_hgraph(tt, ofname);
    break;
  case CNV_FIB_SPMAT:
    __convert_fib_mat(tt, mode, ofname);
    break;
  default:
    fprintf(stderr, "SPLATT ERROR: convert type not implemented.\n");
    exit(1);
  }

  timer_stop(&timers[TIMER_CONVERT]);
  tt_free(tt);
}

