
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

#include "util.h"
#include "csf.h"


/******************************************************************************
 * TYPES
 *****************************************************************************/

/**
* @brief Represents a set with a known (and reasonable) maximum value.
*/
typedef struct
{
  wgt_t * counts; /** The number of times an element was updated. */
  vtx_t * seen;   /** The (unsorted) list of elements that have been seen. */
  vtx_t nseen;    /** The length of seen[]. */
} adj_set;



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Allocate/initialize a set.
*
* @param set The set to allocate.
* @param max_size The maximum element in the set. 2x this memory is allocated.
*/
static void p_set_init(
    adj_set * set,
    vtx_t max_size)
{
  set->counts = calloc(max_size, sizeof(*(set->counts)));
  set->seen = calloc(max_size, sizeof(*(set->seen)));
  set->nseen = 0;
}


/**
* @brief Free all memory allocated for a set.
*
* @param set The set to free.
*/
static void p_set_free(
    adj_set * set)
{
  set->nseen = 0;
  free(set->counts);
  free(set->seen);
}


/**
* @brief Remove (but do not free) all elements from a set. This runs in
*        O(nseen) time.
*
* @param set the set to clear.
*/
static void p_set_clear(
    adj_set * set)
{
  wgt_t * const counts = set->counts;
  vtx_t * const seen = set->seen;
  for(vtx_t i=0; i < set->nseen; ++i) {
    counts[seen[i]] = 0;
    seen[i] = 0;
  }

  set->nseen = 0;
}


/**
* @brief Add a new element to the set or update its count.
*
* @param set The set to modify.
* @param vid The id of the element.
* @param upd How much to modify counts[] by.
*/
static void p_set_update(
    adj_set * set,
    vtx_t vid,
    wgt_t upd)
{
  /* add to set if necessary */
  if(set->counts[vid] == 0) {
    set->seen[set->nseen] = vid;
    set->nseen += 1;
  }

  /* update count */
  set->counts[vid] += upd;
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


static adj_t p_count_adj_size(
    splatt_csf * const csf)
{
  adj_t ncon = 0;

  assert(csf->ntiles == 1);
  csf_sparsity * pt = csf->pt;
  vtx_t const nvtxs = pt->nfibs[0];

  /* type better be big enough */
  assert((idx_t) nvtxs == (vtx_t) nvtxs);

  adj_set set;
  p_set_init(&set, csf->dims[argmax_elem(csf->dims, csf->nmodes)]);

  idx_t parent_start = 0;
  idx_t parent_end = 0;
  for(vtx_t v=0; v < nvtxs; ++v) {
    parent_start = v;
    parent_end = v+1;

    for(idx_t d=1; d < csf->nmodes; ++d) {
      idx_t const start = pt->fptr[d-1][parent_start];
      idx_t const end = pt->fptr[d-1][parent_end];

      idx_t const * const fids = pt->fids[d];
      for(idx_t f=start; f < end; ++f) {
        p_set_update(&set, fids[f], 1);
      }

      ncon += set.nseen;

      /* prepare for next level in the tree */
      parent_start = start;
      parent_end = end;

      p_set_clear(&set);
    }
  }

  p_set_free(&set);

  return ncon;
}

static idx_t p_calc_offset(
    splatt_csf const * const csf,
    idx_t const depth)
{
  idx_t const mode = csf->dim_perm[depth];
  idx_t offset = 0;
  for(idx_t m=0; m < mode; ++m) {
    offset += csf->dims[m];
  }
  return offset;
}

static wgt_t p_count_nnz(
    idx_t * * fptr,
    idx_t const nmodes,
    idx_t depth,
    idx_t const fiber)
{
  if(depth == nmodes-1) {
    return 1;
  }

  idx_t left = fptr[depth][fiber];
  idx_t right = fptr[depth][fiber+1];
  ++depth;

  for(; depth < nmodes-1; ++depth) {
    left = fptr[depth][left];
    right = fptr[depth][right];
  }

  return right - left;
}


static void p_fill_ijk_graph(
    splatt_csf const * const csf,
    splatt_graph * graph)
{
  csf_sparsity * pt = csf->pt;
  vtx_t const nvtxs = graph->nvtxs;

  adj_set set;
  p_set_init(&set, csf->dims[argmax_elem(csf->dims, csf->nmodes)]);

  /* pointing into eind */
  adj_t ncon = 0;

  /* start/end of my subtree */
  idx_t parent_start;
  idx_t parent_end;

  for(vtx_t v=0; v < nvtxs; ++v) {
    parent_start = v;
    parent_end = v+1;

    graph->eptr[v] = ncon;

    for(idx_t d=1; d < csf->nmodes; ++d) {
      idx_t const start = pt->fptr[d-1][parent_start];
      idx_t const end = pt->fptr[d-1][parent_end];

      /* compute adjacency info */
      idx_t const * const fids = pt->fids[d];
      for(idx_t f=start; f < end; ++f) {
        p_set_update(&set, fids[f], p_count_nnz(pt->fptr, csf->nmodes, d, f));
      }

      /* things break if vtx size isn't our sorting size... */
      if(sizeof(*(set.seen)) == sizeof(splatt_idx_t)) {
        quicksort((idx_t *) set.seen, set.nseen);
      }

      /* fill in graph->eind */
      idx_t const id_offset = p_calc_offset(csf, d);
      for(vtx_t e=0; e < set.nseen; ++e) {
        graph->eind[ncon] = set.seen[e] + id_offset;
        if(graph->ewgts != NULL) {
          graph->ewgts[ncon] = set.counts[set.seen[e]];
        }
        ++ncon;
      }

      /* prepare for next level in the tree */
      parent_start = start;
      parent_end = end;

      p_set_clear(&set);
    }
  }

  p_set_free(&set);

  graph->eptr[nvtxs] = graph->nedges;
}


static splatt_graph * p_merge_graphs(
    splatt_graph * * graphs,
    idx_t const ngraphs)
{
  /* count total size */
  vtx_t nvtxs = 0;
  adj_t ncon = 0;
  for(idx_t m=0; m < ngraphs; ++m) {
    nvtxs += graphs[m]->nvtxs;
    ncon += graphs[m]->nedges;
  }

  splatt_graph * ret = graph_alloc(nvtxs, ncon, 0, 1);

  /* fill in ret */
  vtx_t voffset = 0;
  adj_t eoffset = 0;
  for(idx_t m=0; m < ngraphs; ++m) {
    for(vtx_t v=0; v < graphs[m]->nvtxs; ++v) {

      vtx_t const * const eptr = graphs[m]->eptr;
      adj_t const * const eind = graphs[m]->eind;
      wgt_t const * const ewgts = graphs[m]->ewgts;

      ret->eptr[v + voffset] = eptr[v] + eoffset;
      for(adj_t e=eptr[v]; e < eptr[v+1]; ++e) {
        ret->eind[e + eoffset] = eind[e];
        ret->ewgts[e + eoffset] = ewgts[e];
      }
    }
    voffset += graphs[m]->nvtxs;
    eoffset += graphs[m]->nedges;
  }

  return ret;
}


static void __convert_ijk_graph(
  sptensor_t * const tt,
  char const * const ofname)
{
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

  splatt_graph * graphs[MAX_NMODES];

  splatt_csf csf;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    csf_alloc_mode(tt, CSF_INORDER_MINUSONE, m, &csf, opts);

    /* count size of adjacency list */
    adj_t const ncon = p_count_adj_size(&csf);

    graphs[m] = graph_alloc(tt->dims[m], ncon, 0, 1);
    p_fill_ijk_graph(&csf, graphs[m]);

    csf_free_mode(&csf);
  }

  /* merge graphs and write */
  splatt_graph * full_graph = p_merge_graphs(graphs, tt->nmodes);
  FILE * fout = open_f(ofname, "w");
  graph_write_file(full_graph, fout);
  fclose(fout);

  /* clean up */
  graph_free(full_graph);
  splatt_free_opts(opts);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    graph_free(graphs[m]);
  }
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

