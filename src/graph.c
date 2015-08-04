

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "graph.h"

#ifdef SPLATT_USE_PATOH
#include <patoh.h>
#endif


/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/

/**
* @brief Fill the vertex weights array.
*
* @param ft The CSF tensor to derive vertex weights from.
* @param hg The hypegraph structure to modify.
* @param which Vertex weight model to follow, see graph.h.
*/
static void __fill_vwts(
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
    hg->vwts = (idx_t *) malloc(hg->nvtxs * sizeof(idx_t));
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
static idx_t __map_idx(
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
  sptensor_t const * const tt)
{
  hgraph_t * hg = (hgraph_t *) malloc(sizeof(hgraph_t));
  hg->nvtxs = tt->nnz;
  __fill_vwts(NULL, hg, VTX_WT_NONE);

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
  hg->eind = (idx_t *) malloc(tt->nnz * tt->nmodes * sizeof(idx_t));
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
  hgraph_t * hg = (hgraph_t *) malloc(sizeof(hgraph_t));

  /* vertex weights are nnz per fiber */
  hg->nvtxs = ft->nfibs;
  __fill_vwts(ft, hg, VTX_WT_FIB_NNZ);

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
    eptr[1+__map_idx(s, 0, ft)] += ft->sptr[s+1] - ft->sptr[s];

    for(idx_t f=ft->sptr[s]; f < ft->sptr[s+1]; ++f) {
      /* fiber makes another connection with fid */
      eptr[1+__map_idx(ft->fids[f], 1, ft)] += 1;

      /* each nnz now has a contribution too */
      for(idx_t jj=ft->fptr[f]; jj < ft->fptr[f+1]; ++jj) {
        eptr[1+__map_idx(ft->inds[jj], 2, ft)] += 1;
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

  hg->eind = (idx_t *) malloc(ncon * sizeof(idx_t));
  idx_t * const restrict eind = hg->eind;

  /* now fill in eind while using eptr as a marker */
  for(idx_t s=0; s < ft->nslcs; ++s) {
    idx_t const sid = __map_idx(s, 0, ft);
    for(idx_t f = ft->sptr[s]; f < ft->sptr[s+1]; ++f) {
      idx_t const fid = __map_idx(ft->fids[f], 1, ft);
      eind[eptr[1+sid]++] = f;
      eind[eptr[1+fid]++] = f;
      for(idx_t jj=ft->fptr[f]; jj < ft->fptr[f+1]; ++jj) {
        idx_t const nid = __map_idx(ft->inds[jj], 2, ft);
        eind[eptr[1+nid]++] = f;
      }
    }
  }

  return hg;
}


idx_t * hgraph_uncut(
  hgraph_t const * const hg,
  idx_t const * const parts,
  idx_t * const ret_ncut)
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
  *ret_ncut = ncut;

  /* go back and fill in uncut edges */
  idx_t * cut = (idx_t *) malloc(ncut * sizeof(idx_t));
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
  int * vwts = (int *) malloc(nvtxs * sizeof(int));
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
    hwts = (int *) malloc(nnets * sizeof(int));
    for(int h=0; h < nnets; ++h) {
      hwts[h] = (int) hg->hewts[h];
    }
  }

  /* net start/end */
  int * eptr = (int *) malloc((nnets+1) * sizeof(int));
  for(int v=0; v <= nnets; ++v) {
    eptr[v] = (int) hg->eptr[v];
  }

  /* netted vertices */
  int * eind = (int *) malloc(eptr[nnets] * sizeof(int));
  for(int v=0; v < eptr[nnets]; ++v) {
    eind[v] = (int) hg->eind[v];
  }

  int * pvec = (int *) malloc(nvtxs * sizeof(int));
  int * pwts = (int *) malloc(nparts * sizeof(int));
  int cut;

  args._k = (int) nparts;
  PaToH_Alloc(&args, nvtxs, nnets, ncon, vwts, hwts, eptr, eind);

  /* do the partitioning! */
  PaToH_Part(&args, nvtxs, nnets, ncon, 0, vwts, hwts, eptr, eind, NULL, pvec,
      pwts, &cut);

  /* copy patoh output to idx_t */
  idx_t * parts = (idx_t *) malloc(nvtxs * sizeof(idx_t));
  for(idx_t p=0; p < nvtxs; ++p) {
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



