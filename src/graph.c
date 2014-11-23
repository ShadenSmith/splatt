

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
  for(idx_t m=0; m < ft->nmodes; ++m) {
    idx_t pm = ft->dim_perms[mode][m];
    emaps[m] = (int *) malloc(ft->dims[pm] * sizeof(int));
    memset(emaps[m], 0, ft->dims[pm]);

    idx_t h = 0;
    for(idx_t s=0; s < ft->dims[pm]; ++s) {
      /* if slice is non-empty */
      if(ft->sptr[pm][s] != ft->sptr[m][s+1]) {
        emaps[m][s] = h++;
        ++(hg->nhedges);
      } else {
        emaps[m][s] = -1;
      }
    }
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

  /* count hedges and map ind to hedge - this is necessary because empty
   * slices are possible */
  int * emaps[MAX_NMODES];
  __fill_emap(ft, hg, mode, emaps);
  printf("found %d vtxs and %d hedges\n", hg->nvtxs, hg->nhedges);

  /* a) each nnz induces a hyperedge connection
     b) each non-fiber mode accounts for a hyperedge connection */
  idx_t neind = ft->nnz + ((ft->nmodes-1) * hg->nvtxs);
  hg->eptr = (int *) malloc((hg->nhedges+1) * sizeof(int));
  hg->eind = (int *) malloc(neind * sizeof(int));
  hg->eptr[0] = 0;

  /* fill in hedges */
  idx_t vtx = 0;
  for(idx_t s=0; s < ft->dims[mode]; ++s) {
    for(idx_t f = ft->sptr[mode][s]; f < ft->sptr[mode][s+1]; ++f) {
      for(idx_t jj= ft->fptr[mode][f]; jj < ft->fptr[mode][f+1]; ++jj) {

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
  free(hg);
}

