

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "graph.h"



/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/
static void __fill_emap(
  ftensor_t const * const ft,
  hgraph_t * const hg,
  idx_t const mode,
  int ** emaps)
{
  hg->nhedges = 0;
  int h = 0;
  for(idx_t m=0; m < ft->nmodes; ++m) {
    idx_t pm = ft->dim_perms[mode][m];
    emaps[m] = (int *) malloc(ft->dims[pm] * sizeof(int));
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
  int ** emaps)
{
  for(idx_t m=0; m < ft->nmodes; ++m) {
    emaps[m] = NULL;
  }

  hg->nhedges = 0;
  idx_t const pm = ft->dim_perms[mode][2];
  emaps[2] = (int *) malloc(ft->dims[pm] * sizeof(int));
  memset(emaps[2], 0, ft->dims[pm]);
  int h = 0;
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

static void __fill_vwts(
  ftensor_t const * const ft,
  hgraph_t * const hg,
  idx_t const mode)
{
  hg->vwts = (int *) malloc(hg->nvtxs * sizeof(int));
  for(int v=0; v < hg->nvtxs; ++v) {
    hg->vwts[v] = ft->fptr[mode][v+1] - ft->fptr[mode][v];
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
  hg->nvtxs = ft->nfibs[mode];
  hg->vwts = NULL;
  hg->hewts = NULL;

  /* vertex weights are nnz per fiber */
  __fill_vwts(ft, hg, mode);

  /* count hedges and map ind to hedge - this is necessary because empty
   * slices are possible */
  int * emaps[MAX_NMODES];
  __fill_emap(ft, hg, mode, emaps);

  /* a) each nnz induces a hyperedge connection
     b) each non-fiber mode accounts for a hyperedge connection */
  idx_t neind = ft->nnz + ((ft->nmodes-1) * hg->nvtxs);
  hg->eptr = (int *) malloc((hg->nhedges+1) * sizeof(int));
  memset(hg->eptr, 0, (hg->nhedges+1) * sizeof(int));
  hg->eind = (int *) malloc(neind * sizeof(int));

  /* fill in eptr - all offset by 1 to do a prefix sum later */
  for(idx_t s=0; s < ft->dims[mode]; ++s) {
    /* slice hyperedge */
    int hs = emaps[0][s];
    hg->eptr[hs+1] = ft->sptr[mode][s+1] - ft->sptr[mode][s];
    for(idx_t f = ft->sptr[mode][s]; f < ft->sptr[mode][s+1]; ++f) {
      int hfid = emaps[1][ft->fids[mode][f]];
      hg->eptr[hfid+1] += 1;
      for(idx_t jj= ft->fptr[mode][f]; jj < ft->fptr[mode][f+1]; ++jj) {
        int hjj = emaps[2][ft->inds[mode][jj]];
        hg->eptr[hjj+1] += 1;
      }
    }
  }

  /* do a shifted prefix sum to get eptr */
  int saved = hg->eptr[1];
  hg->eptr[1] = 0;
  for(int i=2; i <= hg->nhedges; ++i) {
    int tmp = hg->eptr[i];
    hg->eptr[i] = hg->eptr[i-1] + saved;
    saved = tmp;
  }

  /* now fill in eind while using eptr as a marker */
  int vtx = 0;
  for(idx_t s=0; s < ft->dims[mode]; ++s) {
    int hs = emaps[0][s];
    for(idx_t f = ft->sptr[mode][s]; f < ft->sptr[mode][s+1]; ++f) {
      int hfid = emaps[1][ft->fids[mode][f]];
      hg->eind[hg->eptr[hs+1]++]   = vtx;
      hg->eind[hg->eptr[hfid+1]++] = vtx;
      for(idx_t jj= ft->fptr[mode][f]; jj < ft->fptr[mode][f+1]; ++jj) {
        int hjj = emaps[2][ft->inds[mode][jj]];
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

void hgraph_free(
  hgraph_t * hg)
{
  free(hg->eptr);
  free(hg->eind);
  free(hg->vwts);
  free(hg->hewts);
  free(hg);
}

