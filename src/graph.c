

/******************************************************************************
 * INCLUDES
 *****************************************************************************/


#include "base.h"
#include "graph.h"

#include "csf.h"
#include "sort.h"
#include "util.h"

#ifdef SPLATT_USE_PATOH
#include <patoh.h>
#endif

#ifdef SPLATT_USE_ASHADO
#include <ashado.h>
#endif

#ifdef SPLATT_USE_METIS
/* don't let metis types conflict with splatt */
#define idx_t metis_idx_t
#include <metis.h>
#undef idx_t
#endif


/* use multi-constraint balancing for m-partite graphs */
#ifndef SPLATT_USE_VTX_WGTS
#define SPLATT_USE_VTX_WGTS 0
#endif


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
* @brief Count the number of edges (i.e., the size of adjacency list) of a
*        sparse tensor converted to m-partite graph.
*
* @param csf The tensor to convert.
*
* @return The number of edges.
*/
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


/**
* @brief Compute the offset of a certain CSF tree depth (when all indices are
*        mapped to vertices). This accounts for CSF mode permutation.
*
*        For example, with no permutation and depth=2, this returns
*        csf->dims[0] + csf->dims[1].
*
* @param csf The tensor to use for calculation.
* @param depth The depth to work on.
*
* @return The offset.
*/
static idx_t p_calc_offset(
    splatt_csf const * const csf,
    idx_t const depth)
{
  idx_t const mode = csf_depth_to_mode(csf, depth);
  idx_t offset = 0;
  for(idx_t m=0; m < mode; ++m) {
    offset += csf->dims[m];
  }
  return offset;
}


/**
* @brief Count the nonzeros below a given node in a CSF tensor.
*
* @param fptr The adjacency pointer of the CSF tensor.
* @param nmodes The number of modes in the tensor.
* @param depth The depth of the node
* @param fiber The id of the node.
*
* @return The nonzeros below fptr[depth][fiber].
*/
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


/**
* @brief Fill the contents of a splatt_graph. The graph must already be
*        allocated!
*
* @param csf The tensor to convert.
* @param graph The graph to fill, ALREADY ALLOCATED!
*/
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



/**
* @brief Fill the multi-constraint vertex weights with the #nnz that appear in
*        each index.
*
* @param graph The graph to fill.
* @param tt The tensor we are converting.
*/
static void p_fill_graph_vwgts(
    splatt_graph * const graph,
    splatt_coo const * const tt)
{
  idx_t const nnz = tt->nnz;

  assert(graph->nvwgts == tt->nmodes);

  wgt_t * const vwgts = graph->vwgts;
  memset(vwgts, 0, graph->nvtxs * graph->nvwgts * sizeof(*vwgts));

  idx_t offset = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t const * const inds = tt->ind[m];

    /* each nnz appearance is 1 weight */
    for(idx_t x=0; x < nnz; ++x) {
      idx_t const v = inds[x] + offset;
      vwgts[m + (v * graph->nvwgts)] += 1;
    }
    offset += tt->dims[m];
  }
}

/**
* @brief Takes a list of graphs and returns them stacked on top of each other.
*        No adjacency lists are altered, only vertices added.
*
* @param graphs The graphs to merge.
* @param ngraphs The number of graphs.
*
* @return All graphs stacked.
*/
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

  splatt_graph * ret = graph_alloc(nvtxs, ncon, graphs[0]->nvwgts,
      graphs[0]->ewgts != NULL);

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
        if(ret->ewgts != NULL) {
          ret->ewgts[e + eoffset] = ewgts[e];
        }
      }
    }
    voffset += graphs[m]->nvtxs;
    eoffset += graphs[m]->nedges;
  }

  return ret;
}



/**
* @brief Fill the vertex weights array.
*
* @param ft The CSF tensor to derive vertex weights from.
* @param hg The hypegraph structure to modify.
* @param which Vertex weight model to follow, see graph.h.
*/
static void p_fill_vwts(
  ftensor_t const * const ft,
  hgraph_t * const hg,
  hgraph_vwt_type const which)
{
  switch(which) {
  case VTX_WT_NONE:
    hg->vwts = NULL;
    break;

  /* weight based on nnz in fiber */
  case VTX_WT_FIB_NNZ:
    hg->vwts = (idx_t *) splatt_malloc(hg->nvtxs * sizeof(idx_t));
    #pragma omp parallel for
    for(idx_t v=0; v < hg->nvtxs; ++v) {
      hg->vwts[v] = ft->fptr[v+1] - ft->fptr[v];
    }
  }
}


/**
* @brief Maps an index in a mode of a permuted CSF tensor to a global vertex
*        index. This accounts for the mode permutation using the CSF dim-perm.
*
* @param id The index we are converting (local to the mode).
* @param mode The mode the index lies in (LOCAL TO THE CSF TENSOR).
*             EXAMPLE: a 3 mode tensor would use mode-0 to represent slices,
*             mode-1 to represent fids, and mode-2 to represent the fiber nnz
* @param ft The CSF tensor with dim_perm.
*
* @return 'id', converted to global vertex indices. EXAMPLE: k -> (I+J+k).
*/
static idx_t p_map_idx(
  idx_t id,
  idx_t const mode,
  ftensor_t const * const ft)
{
  idx_t m = 0;
  while(m != ft->dim_perm[mode]) {
    id += ft->dims[m++];
  }
  return id;
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

hgraph_t * hgraph_nnz_alloc(
  splatt_coo const * const tt)
{
  hgraph_t * hg = (hgraph_t *) splatt_malloc(sizeof(hgraph_t));
  hg->nvtxs = tt->nnz;
  p_fill_vwts(NULL, hg, VTX_WT_NONE);

  /* # hyper-edges = I + J + K + ... */
  hg->hewts = NULL;
  hg->nhedges = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    hg->nhedges += tt->dims[m];
  }

  /* fill in eptr shifted by 1 index. */
  hg->eptr = (idx_t *) calloc(hg->nhedges+1, sizeof(idx_t));
  idx_t * const restrict eptr = hg->eptr;
  idx_t offset = 1;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t const * const restrict ind = tt->ind[m];
    for(idx_t n=0; n < tt->nnz; ++n) {
      eptr[offset+ind[n]] += 1;
    }
    offset += tt->dims[m];
  }

  /* do a shifted prefix sum to get eptr */
  idx_t saved = eptr[1];
  eptr[1] = 0;
  for(idx_t i=2; i <= hg->nhedges; ++i) {
    idx_t tmp = eptr[i];
    eptr[i] = eptr[i-1] + saved;
    saved = tmp;
  }

  /* each nnz causes 'nmodes' connections */
  hg->eind = (idx_t *) splatt_malloc(tt->nnz * tt->nmodes * sizeof(idx_t));
  idx_t * const restrict eind = hg->eind;

  offset = 1;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t const * const restrict ind = tt->ind[m];
    for(idx_t n=0; n < tt->nnz; ++n) {
      eind[eptr[offset+ind[n]]++] = n;
    }
    offset += tt->dims[m];
  }

  assert(eptr[hg->nhedges] == tt->nnz * tt->nmodes);
  return hg;
}



hgraph_t * hgraph_fib_alloc(
  ftensor_t const * const ft,
  idx_t const mode)
{
  hgraph_t * hg = (hgraph_t *) splatt_malloc(sizeof(hgraph_t));

  /* vertex weights are nnz per fiber */
  hg->nvtxs = ft->nfibs;
  p_fill_vwts(ft, hg, VTX_WT_FIB_NNZ);

  /* # hyper-edges = I + J + K + ... */
  hg->hewts = NULL;
  hg->nhedges = 0;
  for(idx_t m=0; m < ft->nmodes; ++m) {
    hg->nhedges += ft->dims[m];
  }

  /* fill in eptr shifted by 1 idx:
   *   a) each nnz induces a hyperedge connection
   *   b) each non-fiber mode accounts for a hyperedge connection
   */
  hg->eptr = (idx_t *) calloc(hg->nhedges+1, sizeof(idx_t));
  idx_t * const restrict eptr = hg->eptr;
  for(idx_t s=0; s < ft->nslcs; ++s) {
    /* the slice hyperedge has nfibers more connections */
    eptr[1+p_map_idx(s, 0, ft)] += ft->sptr[s+1] - ft->sptr[s];

    for(idx_t f=ft->sptr[s]; f < ft->sptr[s+1]; ++f) {
      /* fiber makes another connection with fid */
      eptr[1+p_map_idx(ft->fids[f], 1, ft)] += 1;

      /* each nnz now has a contribution too */
      for(idx_t jj=ft->fptr[f]; jj < ft->fptr[f+1]; ++jj) {
        eptr[1+p_map_idx(ft->inds[jj], 2, ft)] += 1;
      }
    }
  }

  /* do a shifted prefix sum to get eptr */
  idx_t ncon = eptr[1];
  idx_t saved = eptr[1];
  eptr[1] = 0;
  for(idx_t i=2; i <= hg->nhedges; ++i) {
    ncon += eptr[i];
    idx_t tmp = eptr[i];
    eptr[i] = eptr[i-1] + saved;
    saved = tmp;
  }

  hg->eind = (idx_t *) splatt_malloc(ncon * sizeof(idx_t));
  idx_t * const restrict eind = hg->eind;

  /* now fill in eind while using eptr as a marker */
  for(idx_t s=0; s < ft->nslcs; ++s) {
    idx_t const sid = p_map_idx(s, 0, ft);
    for(idx_t f = ft->sptr[s]; f < ft->sptr[s+1]; ++f) {
      idx_t const fid = p_map_idx(ft->fids[f], 1, ft);
      eind[eptr[1+sid]++] = f;
      eind[eptr[1+fid]++] = f;
      for(idx_t jj=ft->fptr[f]; jj < ft->fptr[f+1]; ++jj) {
        idx_t const nid = p_map_idx(ft->inds[jj], 2, ft);
        eind[eptr[1+nid]++] = f;
      }
    }
  }

  return hg;
}


idx_t * hgraph_uncut(
  hgraph_t const * const hg,
  idx_t const * const parts,
  idx_t * const ret_nnotcut)
{
  idx_t const nhedges = (idx_t) hg->nhedges;
  idx_t const nvtxs = (idx_t)hg->nvtxs;

  idx_t const * const eptr = hg->eptr;
  idx_t const * const eind = hg->eind;

  idx_t ncut = 0;
  for(idx_t h=0; h < nhedges; ++h) {
    int iscut = 0;
    idx_t const firstpart = parts[eind[eptr[h]]];
    for(idx_t e=eptr[h]+1; e < eptr[h+1]; ++e) {
      idx_t const vtx = eind[e];
      if(parts[vtx] != firstpart) {
        iscut = 1;
        break;
      }
    }
    if(iscut == 0) {
      ++ncut;
    }
  }
  *ret_nnotcut = ncut;

  /* go back and fill in uncut edges */
  idx_t * cut = (idx_t *) splatt_malloc(ncut * sizeof(idx_t));
  idx_t ptr = 0;
  for(idx_t h=0; h < nhedges; ++h) {
    int iscut = 0;
    idx_t const firstpart = parts[eind[eptr[h]]];
    for(idx_t e=eptr[h]+1; e < eptr[h+1]; ++e) {
      idx_t const vtx = eind[e];
      if(parts[vtx] != firstpart) {
        iscut = 1;
        break;
      }
    }
    if(iscut == 0) {
      cut[ptr++] = h;
    }
  }

  return cut;
}


void hgraph_free(
  hgraph_t * hg)
{
  free(hg->eptr);
  free(hg->eind);
  free(hg->vwts);
  free(hg->hewts);
  free(hg);
}


splatt_graph * graph_convert(
    splatt_coo * const tt)
{
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

  splatt_graph * graphs[MAX_NMODES];

  splatt_csf csf;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    csf_alloc_mode(tt, CSF_INORDER_MINUSONE, m, &csf, opts);

    /* count size of adjacency list */
    adj_t const ncon = p_count_adj_size(&csf);

#if SPLATT_USE_VTX_WGTS == 0
    graphs[m] = graph_alloc(tt->dims[m], ncon, 0, 1);
#else
    graphs[m] = graph_alloc(tt->dims[m], ncon, tt->nmodes, 1);
#endif
    p_fill_ijk_graph(&csf, graphs[m]);

    csf_free_mode(&csf);
  }

  /* merge graphs and write */
  splatt_graph * full_graph = p_merge_graphs(graphs, tt->nmodes);

  /* cleanup */
  splatt_free_opts(opts);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    graph_free(graphs[m]);
  }

  /* handle vertex weights */
  if(full_graph->nvwgts > 0) {
    p_fill_graph_vwgts(full_graph, tt);
  }

  return full_graph;
}


splatt_graph * graph_alloc(
    vtx_t nvtxs,
    adj_t nedges,
    int num_vtx_wgts,
    int use_edge_wgts)
{
  splatt_graph * ret = splatt_malloc(sizeof(*ret));

  ret->nvtxs = nvtxs;
  ret->nedges = nedges;
  ret->eptr = splatt_malloc((nvtxs+1) * sizeof(*(ret->eptr)));
  ret->eind = splatt_malloc(nedges * sizeof(*(ret->eind)));

  ret->eptr[nvtxs] = nedges;

  ret->nvwgts = num_vtx_wgts;
  if(num_vtx_wgts) {
    ret->vwgts = splatt_malloc(nvtxs * ret->nvwgts * sizeof(*(ret->vwgts)));
  } else {
    ret->vwgts = NULL;
  }

  if(use_edge_wgts) {
    ret->ewgts = splatt_malloc(nedges * sizeof(*(ret->ewgts)));
  } else {
    ret->ewgts = NULL;
  }

  return ret;
}


void graph_free(
    splatt_graph * graph)
{
  free(graph->eptr);
  free(graph->eind);
  free(graph->vwgts);
  free(graph->ewgts);
  free(graph);
}



#ifdef SPLATT_USE_PATOH
idx_t * patoh_part(
    hgraph_t const * const hg,
    idx_t const nparts)
{
  PaToH_Parameters args;
  PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_SPEED);

  int const nvtxs = hg->nvtxs;
  int const nnets = hg->nhedges;
  int const ncon = 1;

  /* vertex weights */
  int * vwts = (int *) splatt_malloc(nvtxs * sizeof(int));
  if(hg->vwts != NULL) {
    for(int v=0; v < nvtxs; ++v) {
      vwts[v] = (int) hg->vwts[v];
    }
  } else {
    for(int v=0; v < nvtxs; ++v) {
      vwts[v] = 1;
    }
  }

  /* edge weights */
  int * hwts = NULL;
  if(hg->hewts != NULL) {
    hwts = (int *) splatt_malloc(nnets * sizeof(int));
    for(int h=0; h < nnets; ++h) {
      hwts[h] = (int) hg->hewts[h];
    }
  }

  /* net start/end */
  int * eptr = (int *) splatt_malloc((nnets+1) * sizeof(int));
  for(int v=0; v <= nnets; ++v) {
    eptr[v] = (int) hg->eptr[v];
  }

  /* netted vertices */
  int * eind = (int *) splatt_malloc(eptr[nnets] * sizeof(int));
  for(int v=0; v < eptr[nnets]; ++v) {
    eind[v] = (int) hg->eind[v];
  }

  int * pvec = (int *) splatt_malloc(nvtxs * sizeof(int));
  int * pwts = (int *) splatt_malloc(nparts * sizeof(int));
  int cut;

  args._k = (int) nparts;
  PaToH_Alloc(&args, nvtxs, nnets, ncon, vwts, hwts, eptr, eind);

  /* do the partitioning! */
  PaToH_Part(&args, nvtxs, nnets, ncon, 0, vwts, hwts, eptr, eind, NULL, pvec,
      pwts, &cut);

  /* copy patoh output to idx_t */
  idx_t * parts = (idx_t *) splatt_malloc(nvtxs * sizeof(idx_t));
  for(idx_t p=0; p < hg->nvtxs; ++p) {
    parts[p] = (idx_t) pvec[p];
  }

  PaToH_Free();
  free(vwts);
  free(hwts);
  free(eptr);
  free(eind);
  free(pvec);
  free(pwts);

  return parts;
}
#endif


#ifdef SPLATT_USE_ASHADO
idx_t * ashado_part(
    hgraph_t const * const hg,
    idx_t const nparts)
{
  double * opts = ashado_default_opts();
  idx_t * part = (idx_t *) splatt_malloc(hg->nvtxs * sizeof(idx_t));

  ashado_partition(nparts, hg->nvtxs, hg->nhedges, hg->eptr, hg->eind,
      hg->vwts, hg->hewts, opts, 5, part);

  free(opts);
  return part;
}
#endif


#ifdef SPLATT_USE_METIS
splatt_idx_t *  metis_part(
    splatt_graph * graph,
    splatt_idx_t const num_partitions,
    splatt_idx_t * edgecut)
{
  metis_idx_t nvtxs = graph->nvtxs;
  metis_idx_t ncon = 1;
  metis_idx_t nparts = num_partitions;
  metis_idx_t cut = 0;

  /* copy the adj structure */
  metis_idx_t * xadj = splatt_malloc((nvtxs+1) * sizeof(*xadj));
  for(metis_idx_t v=0; v <= nvtxs; ++v) {
    xadj[v] = graph->eptr[v];
  }
  metis_idx_t * adjncy = splatt_malloc(xadj[nvtxs] * sizeof(*adjncy));
  for(metis_idx_t e=0; e < xadj[nvtxs]; ++e) {
    adjncy[e] = graph->eind[e];
  }

  /* weights */
  metis_idx_t * vwgt = NULL;
  metis_idx_t * ewgt = NULL;
  if(graph->vwgts != NULL) {
    /* graph number of vertex weights */
    ncon = graph->nvwgts;
    vwgt = splatt_malloc(nvtxs * ncon * sizeof(*vwgt));
    for(metis_idx_t v=0; v < nvtxs * ncon; ++v) {
      vwgt[v] = graph->vwgts[v];
    }
  }
  if(graph->ewgts != NULL) {
    ewgt = splatt_malloc(xadj[nvtxs] * sizeof(*ewgt));
    for(metis_idx_t e=0; e < xadj[nvtxs]; ++e) {
      ewgt[e] = graph->ewgts[e];
    }
  }

  /* allocate partitioning info */
  metis_idx_t * metis_parts = splatt_malloc(nvtxs * sizeof(*metis_parts));

  /* do the partitioning! */
  int ret = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL,
      ewgt, &nparts, NULL, NULL, NULL /* opts */, &cut, metis_parts);
  if(ret != METIS_OK) {
    fprintf(stderr, "METIS_PartGraphRecursive returned %d\n", ret);
  }
  *edgecut = cut;

  /* cleanup */
  splatt_free(xadj);
  splatt_free(adjncy);
  if(graph->vwgts != NULL) {
    splatt_free(vwgt);
  }
  if(graph->ewgts != NULL) {
    splatt_free(ewgt);
  }

  /* copy into splatt_idx_t */
  splatt_idx_t * parts = splatt_malloc(nvtxs * sizeof(*parts));
  for(metis_idx_t v=0; v < nvtxs; ++v) {
    parts[v] = metis_parts[v];
  }
  splatt_free(metis_parts);

  return parts;
}
#endif


