
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
* @brief Key-value pair.
*/
typedef struct
{
  idx_t v;
  unsigned int cnt;
} kvp_t;

static idx_t nreallocs;
static idx_t const ADJ_START_ALLOC = 8;


typedef struct
{
  wgt_t * counts;
  vtx_t * seen;
  vtx_t nseen;
} adj_set;


static void p_set_init(
    adj_set * set,
    vtx_t max_size)
{
  set->counts = calloc(max_size, sizeof(*(set->counts)));
  set->seen = calloc(max_size, sizeof(*(set->seen)));
  set->nseen = 0;
}


static void p_set_free(
    adj_set * set)
{
  set->nseen = 0;
  free(set->counts);
  free(set->seen);
}


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


static void p_fill_mpart_graph(
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
        quicksort(set.seen, set.nseen);
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


static void __convert_mpart_graph(
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
    p_fill_mpart_graph(&csf, graphs[m]);

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
  case CNV_NPART_GRAPH:
    __convert_mpart_graph(tt, ofname);
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

