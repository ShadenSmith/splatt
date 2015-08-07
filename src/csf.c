
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "csf.h"
#include "sort.h"

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

#if 0
static void __print_csf(
  csf_t const * const ft)
{
  printf("-----------\n");
  printf("nmodes: %lu nnz: %lu\n", ft->nmodes, ft->nnz);
  printf("dims: %lu", ft->dims[0]);
  for(idx_t m=1; m < ft->nmodes; ++m) {
    printf("x%lu", ft->dims[m]);
  }
  printf("\n");

  printf("fptr:\n");
  for(idx_t m=0; m < ft->nmodes-1; ++m) {
    printf("[%lu] ", ft->nfibs[m]);
    for(idx_t f=0; f < ft->nfibs[m]; ++f) {
      printf(" (%lu, %lu)", ft->fptr[m][f], ft->fids[m][f]);
    }
    printf(" %lu\n", ft->fptr[m][ft->nfibs[m]]);
  }

  /* vals/inds */
  printf("[%lu] ", ft->nfibs[ft->nmodes-1]);
  for(idx_t f=0; f < ft->nfibs[ft->nmodes-1]; ++f) {
    printf(" %3lu", ft->fids[ft->nmodes-1][f] + 1);
  }
  printf("\n");
  for(idx_t n=0; n < ft->nnz; ++n) {
    printf(" %0.1f", ft->vals[n]);
  }
  printf("\n");

  printf("-----------\n\n");
}
#endif


static void __mk_outerptr(
  csf_t * const ft,
  sptensor_t const * const tt,
  idx_t const mode)
{
  /* the mode after accounting for dim_perm */
  idx_t const mperm = ft->dim_perm[mode];
  idx_t const * const restrict ttind = tt->ind[mperm];

  ft->nfibs[mode] = ft->dims[mperm];
  ft->fptr[mode] = (idx_t *) malloc((ft->nfibs[mode]+1) * sizeof(idx_t));
  idx_t  * const restrict fp = ft->fptr[mode];
  fp[0] = 0;
  idx_t nfound = 1;
  for(idx_t n=1; n < ft->nnz; ++n) {
    /* check for end of outer index */
    if(ttind[n] != ttind[n-1]) {
      fp[nfound++] = n;
    }
  }

  /* account for empty slices? */
  while(nfound <= ft->nfibs[mode]) {
    fp[nfound++] = ft->nnz;
  }
}


static void __mk_fptr(
  csf_t * const ft,
  sptensor_t const * const tt,
  idx_t const mode)
{
  assert(mode < ft->nmodes);

  /* the mode after accounting for dim_perm */
  idx_t const mperm = ft->dim_perm[mode];
  idx_t const * const restrict ttind = tt->ind[mperm];

  /* outer mode is easy; just look at outer indices */
  if(mode == 0) {
    __mk_outerptr(ft, tt, mode);
    return;
  }

  /* we will edit this to point to the new fiber idxs instead of nnz */
  idx_t * const restrict fprev = ft->fptr[mode-1];

  /* first count nfibers */
  ft->nfibs[mode] = 0;
  /* foreach 'slice' in the previous dimension */
  for(idx_t s=0; s < ft->nfibs[mode-1]; ++s) {
    ft->nfibs[mode] += 1; /* one by default */
    /* count fibers in current hyperplane*/
    for(idx_t f=fprev[s]+1; f < fprev[s+1]; ++f) {
      if(ttind[f] != ttind[f-1]) {
        ft->nfibs[mode] += 1;
      }
    }
  }

  ft->fptr[mode] = (idx_t *) malloc((ft->nfibs[mode]+1) * sizeof(idx_t));
  ft->fids[mode] = (idx_t *) malloc(ft->nfibs[mode] * sizeof(idx_t));
  idx_t * const restrict fp = ft->fptr[mode];
  idx_t * const restrict fi = ft->fids[mode];
  fp[0] = 0;

  /* now fill in fiber info */
  idx_t nfound = 0;
  for(idx_t s=0; s < ft->nfibs[mode-1]; ++s) {
    idx_t const start = fprev[s]+1;
    idx_t const end = fprev[s+1];

    /* mark start of subtree */
    fprev[s] = nfound;
    fi[nfound] = ttind[start-1];
    fp[nfound++] = start-1;

    /* mark fibers in current hyperplane */
    for(idx_t f=start; f < end; ++f) {
      if(ttind[f] != ttind[f-1]) {
        fi[nfound] = ttind[f];
        fp[nfound++] = f;
      }
    }
  }

  /* mark end of last hyperplane */
  fprev[ft->nfibs[mode-1]] = ft->nfibs[mode];

  /* account for empty slices? */
  while(nfound <= ft->nfibs[mode]) {
    fp[nfound++] = ft->nnz;
  }
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void csf_alloc(
  csf_t * const ft,
  sptensor_t * const tt,
  idx_t const mode,
  splatt_tile_t which_tile)
{
  idx_t const nmodes = tt->nmodes;
  ft->nmodes = nmodes;
  ft->nnz = tt->nnz;
  ft->fptr = (idx_t **) malloc(tt->nmodes * sizeof(idx_t *));
  ft->fids = (idx_t **) malloc(tt->nmodes * sizeof(idx_t *));
  for(idx_t m=0; m < nmodes; ++m) {
    ft->dims[m] = tt->dims[m];
    ft->dim_perm[m] = m;
    ft->fptr[m] = NULL;
    ft->fids[m] = NULL;
  }

  /* get the indices in order */
  tt_sort(tt, mode, ft->dim_perm);

  /* last row of fptr is just nonzero inds */
  ft->nfibs[nmodes-1] = ft->nnz;
  ft->fids[nmodes-1] = (idx_t *) malloc(ft->nnz * sizeof(idx_t));
  ft->vals           = (val_t *) malloc(ft->nnz * sizeof(val_t));
  memcpy(ft->fids[nmodes-1], tt->ind[nmodes-1], ft->nnz * sizeof(idx_t));
  memcpy(ft->vals, tt->vals, ft->nnz * sizeof(val_t));

  /* create fptr entries for the rest of the modes, working up from */
  for(idx_t m=0; m < tt->nmodes-1; ++m) {
    __mk_fptr(ft, tt, m);
  }

  csf_free(ft);
}


void csf_free(
  csf_t * const ft)
{
  free(ft->vals);
  for(idx_t m=0; m < ft->nmodes; ++m) {
    free(ft->fptr[m]);
    free(ft->fids[m]);
  }
  free(ft->fids);
  free(ft->fptr);
}


