
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "coo.h"
#include "ftensor.h"
#include "graph.h"
#include "io.h"
#include "matrix.h"
#include "convert.h"
#include "stats.h"
#include "timer.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

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
static void p_convert_fib_hgraph(
  splatt_coo * tt,
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
static void p_convert_nnz_hgraph(
  splatt_coo const * const tt,
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
static void p_convert_fib_mat(
  splatt_coo * tt,
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




static void p_convert_ijk_graph(
  splatt_coo * const tt,
  char const * const ofname)
{
  /* convert to graph */
  splatt_graph * graph = graph_convert(tt);

  FILE * fout = open_f(ofname, "w");

  graph_write_file(graph, fout);

  fclose(fout);
  graph_free(graph);
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
  splatt_coo * tt = tt_read(ifname);
  if(tt == NULL) {
    return;
  }
  stats_tt(tt, ifname, STATS_BASIC, 0, NULL);

  timer_start(&timers[TIMER_CONVERT]);

  switch(type) {
  case CNV_IJK_GRAPH:
    p_convert_ijk_graph(tt, ofname);
    break;
  case CNV_FIB_HGRAPH:
    p_convert_fib_hgraph(tt, mode, ofname);
    break;
  case CNV_NNZ_HGRAPH:
    p_convert_nnz_hgraph(tt, ofname);
    break;
  case CNV_FIB_SPMAT:
    p_convert_fib_mat(tt, mode, ofname);
    break;
  case CNV_BINARY:
    tt_write_binary(tt, ofname);
    break;
  case CNV_COORD:
    tt_write(tt, ofname);
    break;
  default:
    fprintf(stderr, "SPLATT ERROR: convert type not implemented.\n");
    exit(1);
  }

  timer_stop(&timers[TIMER_CONVERT]);
  tt_free(tt);
}

