

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "graph.h"



/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/
#if 0
static void __fill_emap(
  ftensor_t ** ft,
  hgraph_t * const hg,
  idx_t const mode,
  idx_t ** emaps)
{
  hg->nhedges = 0;
  idx_t h = 0;
  idx_t const nmodes = ft[0]->nmodes;
  for(idx_t m=0; m < nmodes; ++m) {
    idx_t pm = ft->dim_perms[mode][m];
    emaps[m] = (idx_t *) malloc(ft->dims[pm] * sizeof(idx_t));
    memset(emaps[m], 0, ft->dims[pm]);

    for(idx_t s=0; s < ft->dims[pm]; ++s) {
      /* if slice is non-empty */
      if(ft->sptr[pm][s] != ft->sptr[pm][s+1]) {
        emaps[m][s] = h++;
        ++(hg->nhedges);
      } else {
        emaps[m][s] = -1;
      }
    }
  }
}

static void __fill_emap_fibonly(
  ftensor_t const * const ft,
  hgraph_t * const hg,
  idx_t const mode,
  idx_t ** emaps)
{
  for(idx_t m=0; m < ft->nmodes; ++m) {
    emaps[m] = NULL;
  }

  hg->nhedges = 0;
  idx_t const pm = ft->dim_perms[mode][2];
  emaps[2] = (idx_t *) malloc(ft->dims[pm] * sizeof(idx_t));
  memset(emaps[2], 0, ft->dims[pm]);
  idx_t h = 0;
  for(idx_t s=0; s < ft->dims[pm]; ++s) {
    /* if slice is non-empty */
    if(ft->sptr[pm][s] != ft->sptr[pm][s+1]) {
      emaps[2][s] = h++;
      ++(hg->nhedges);
    } else {
      emaps[2][s] = -1;
    }
  }
}
#endif

static void __fill_vwts(
  ftensor_t const * const ft,
  hgraph_t * const hg)
{
  hg->vwts = (idx_t *) malloc(hg->nvtxs * sizeof(idx_t));
  for(idx_t v=0; v < hg->nvtxs; ++v) {
    hg->vwts[v] = ft->fptr[v+1] - ft->fptr[v];
  }
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
hgraph_t * hgraph_fib_alloc(
  ftensor_t const * const ft,
  idx_t const mode)
{
  hgraph_t * hg = (hgraph_t *) malloc(sizeof(hgraph_t));
  hg->nvtxs = ft->nfibs;
  hg->vwts = NULL;
  hg->hewts = NULL;

  /* vertex weights are nnz per fiber */
  __fill_vwts(ft, hg);

  /* count hedges and map ind to hedge - this is necessary because empty
   * slices are possible */
  idx_t * emaps[MAX_NMODES];
#if 0
  /* XXX: TODO */
  __fill_emap(ft, hg, mode, emaps);
#endif

  /* a) each nnz induces a hyperedge connection
     b) each non-fiber mode accounts for a hyperedge connection */
  idx_t neind = ft->nnz + ((ft->nmodes-1) * hg->nvtxs);
  hg->eptr = (idx_t *) malloc((hg->nhedges+1) * sizeof(idx_t));
  memset(hg->eptr, 0, (hg->nhedges+1) * sizeof(idx_t));
  hg->eind = (idx_t *) malloc(neind * sizeof(idx_t));

  /* fill in eptr - all offset by 1 to do a prefix sum later */
  for(idx_t s=0; s < ft->dims[mode]; ++s) {
    /* slice hyperedge */
    idx_t hs = emaps[0][s];
    hg->eptr[hs+1] = ft->sptr[s+1] - ft->sptr[s];
    for(idx_t f = ft->sptr[s]; f < ft->sptr[s+1]; ++f) {
      idx_t hfid = emaps[1][ft->fids[f]];
      hg->eptr[hfid+1] += 1;
      for(idx_t jj= ft->fptr[f]; jj < ft->fptr[f+1]; ++jj) {
        idx_t hjj = emaps[2][ft->inds[jj]];
        hg->eptr[hjj+1] += 1;
      }
    }
  }

  /* do a shifted prefix sum to get eptr */
  idx_t saved = hg->eptr[1];
  hg->eptr[1] = 0;
  for(idx_t i=2; i <= hg->nhedges; ++i) {
    idx_t tmp = hg->eptr[i];
    hg->eptr[i] = hg->eptr[i-1] + saved;
    saved = tmp;
  }

  /* now fill in eind while using eptr as a marker */
  idx_t vtx = 0;
  for(idx_t s=0; s < ft->dims[mode]; ++s) {
    idx_t hs = emaps[0][s];
    for(idx_t f = ft->sptr[s]; f < ft->sptr[s+1]; ++f) {
      idx_t hfid = emaps[1][ft->fids[f]];
      hg->eind[hg->eptr[hs+1]++]   = vtx;
      hg->eind[hg->eptr[hfid+1]++] = vtx;
      for(idx_t jj= ft->fptr[f]; jj < ft->fptr[f+1]; ++jj) {
        idx_t hjj = emaps[2][ft->inds[jj]];
        hg->eind[hg->eptr[hjj+1]++] = vtx;
      }
      ++vtx;
    }
  }

  for(idx_t m=0; m < ft->nmodes; ++m) {
    free(emaps[m]);
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

