

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sort.h"
#include "io.h"
#include "matrix.h"
#include "ftensor.h"
#include "tile.h"
#include "util.h"


/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/
static void __create_fptr(
  ftensor_t * const ft,
  sptensor_t const * const tt,
  idx_t const mode)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;

  /* permuted tt->ind makes things a bit easier */
  idx_t * ttinds[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    ttinds[m] = tt->ind[ft->dim_perm[m]];
  }
  /* this avoids some maybe-uninitialized warnings */
  for(idx_t m=nmodes; m < MAX_NMODES; ++m) {
    ttinds[m] = NULL;
  }

  /* count fibers and copy inds/vals into ft */
  ft->inds[0] = ttinds[nmodes-1][0];
  ft->vals[0] = tt->vals[0];

  /* count fibers in tt */
  idx_t nfibs = 0;
  #pragma omp parallel for reduction(+:nfibs)
  for(idx_t n=1; n < nnz; ++n) {
    for(idx_t m=0; m < nmodes-1; ++m) {
      /* check for new fiber */
      if(ttinds[m][n] != ttinds[m][n-1]) {
        ++nfibs;
        break;
      }
    }
    ft->inds[n] = ttinds[nmodes-1][n];
    ft->vals[n] = tt->vals[n];
  }
  /* account for first fiber (inds[0]) */
  ++nfibs;

  /* allocate fiber structure */
  ft->nfibs = nfibs;
  ft->fptr = (idx_t *) malloc((nfibs+1) * sizeof(idx_t));
  ft->fids = (idx_t *) malloc(nfibs * sizeof(idx_t));
  if(ft->tiled != SPLATT_NOTILE) {
    ft->sids = (idx_t *) malloc(nfibs * sizeof(idx_t));
  }

  /* initialize boundary values */
  ft->fptr[0] = 0;
  ft->fptr[nfibs] = nnz;
  ft->fids[0] = ttinds[1][0];
  if(ft->tiled != SPLATT_NOTILE) {
    ft->sids[0] = ttinds[0][0];
  }

  idx_t fib = 1;
  for(idx_t n=1; n < nnz; ++n) {
    int newfib = 0;
    /* check for new fiber */
    for(idx_t m=0; m < nmodes-1; ++m) {
      if(ttinds[m][n] != ttinds[m][n-1]) {
        newfib = 1;
        break;
      }
    }
    if(newfib) {
      ft->fptr[fib] = n;
      ft->fids[fib] = ttinds[1][n];
      if(ft->tiled != SPLATT_NOTILE) {
        ft->sids[fib] = ttinds[0][n];
      }
      ++fib;
    }
  }
}


static void __create_slabptr(
  ftensor_t * const ft,
  sptensor_t const * const tt,
  idx_t const mode)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;
  idx_t const tsize = TILE_SIZES[0];
  idx_t const nslabs = tt->dims[mode] / tsize + (tt->dims[mode] % tsize != 0);

  ft->nslabs = nslabs;
  ft->slabptr = (idx_t *) malloc((nslabs+1) * sizeof(idx_t));

  idx_t const nfibs = ft->nfibs;
  /* count slices */
  idx_t slices = 1;
  for(idx_t f=1; f < nfibs; ++f) {
    idx_t const slice = ft->sids[f];
    if(ft->sids[f] != ft->sids[f-1]) {
      ++slices;
    }
  }
  idx_t * sptr = (idx_t *) malloc((slices+1) * sizeof(idx_t));
  idx_t * sids = (idx_t *) malloc(slices * sizeof(idx_t));

  sptr[0] = 0;
  sids[0] = ft->sids[0];
  ft->slabptr[0] = 0;
  idx_t s = 1;
  idx_t slab = 1;
  for(idx_t f=1; f < nfibs; ++f) {
    if(ft->sids[f] != ft->sids[f-1]) {
      sptr[s] = f;
      sids[s] = ft->sids[f];
      /* update slabptr if we've moved to the next slab */
      while(sids[s] / tsize > slab-1) {
        ft->slabptr[slab++] = s;
      }
      ++s;
    }
  }

  /* TODO: Why free this for coop, but not indv? - nope, does not work... */
  /* update ft with new data structures */
  free(ft->sids);
  ft->sids = sids;
  ft->sptr = sptr;

  /* account for any empty slabs at end */
  ft->nslabs = slab;
  ft->slabptr[slab] = slices;
  ft->sptr[slices] = nfibs;
}


static void __create_sliceptr(
  ftensor_t * const ft,
  sptensor_t const * const tt,
  idx_t const mode)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;

  idx_t const nslices = ft->dims[mode];
  ft->sptr = (idx_t *) malloc((nslices+1) * sizeof(idx_t));
  ft->nslcs = nslices;

  /* permuted tt->ind makes things a bit easier */
  idx_t * ttinds[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    ttinds[m] = tt->ind[ft->dim_perm[m]];
  }
  /* this avoids some maybe-uninitialized warnings */
  for(idx_t m=nmodes; m < MAX_NMODES; ++m) {
    ttinds[m] = NULL;
  }

  idx_t slice = 0;
  ft->sptr[slice++] = 0;
  while(slice != ttinds[0][0]+1) {
    ft->sptr[slice++] = 0;
  }

  idx_t fib = 1;
  for(idx_t n=1; n < nnz; ++n) {
    int newfib = 0;
    /* check for new fiber */
    for(idx_t m=0; m < nmodes-1; ++m) {
      if(ttinds[m][n] != ttinds[m][n-1]) {
        newfib = 1;
        break;
      }
    }
    if(newfib) {
      /* increment slice if necessary and account for empty slices */
      while(slice != ttinds[0][n]+1) {
        ft->sptr[slice++] = fib;
      }
      ++fib;
    }
  }
  /* account for any empty slices at end */
  for(idx_t s=slice; s <= ft->dims[mode]; ++s) {
    ft->sptr[s] = ft->nfibs;
  }
}


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

int splatt_csf_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_csf_t ** tensors,
    double const * const options)
{
  sptensor_t * tt = tt_read(fname);
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  tt_remove_empty(tt);

  splatt_csf_t * fts = (splatt_csf_t *) malloc(tt->nmodes*sizeof(splatt_csf_t));

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ften_alloc(fts + m, tt, m, (int) options[SPLATT_OPTION_TILE]);
  }

  *tensors = fts;
  *nmodes = tt->nmodes;

  tt_free(tt);

  return SPLATT_SUCCESS;
}

int splatt_csf_convert(
    splatt_idx_t const nmodes,
    splatt_idx_t const nnz,
    splatt_idx_t ** const inds,
    splatt_val_t * const vals,
    splatt_csf_t ** tensors,
    double const * const options)
{
  splatt_csf_t * fts = (splatt_csf_t *) malloc(nmodes * sizeof(splatt_csf_t));

  sptensor_t tt;
  tt_fill(&tt, nnz, nmodes, inds, vals);
  tt_remove_empty(&tt);

  for(idx_t m=0; m < nmodes; ++m) {
    ften_alloc(fts + m, &tt, m, (int) options[SPLATT_OPTION_TILE]);
  }

  *tensors = fts;

  return SPLATT_SUCCESS;
}

void splatt_free_csf(
  splatt_idx_t const nmodes,
  splatt_csf_t ** tensors)
{
  for(idx_t m=0; m < nmodes; ++m) {
    ften_free(tensors[m]);
  }
  free(tensors);
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void ften_alloc(
  ftensor_t * const ft,
  sptensor_t * const tt,
  idx_t const mode,
  int const tile)
{
  ft->nnz = tt->nnz;
  ft->nmodes = tt->nmodes;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ft->dims[m] = tt->dims[m];
  }
  ft->tiled = tt->tiled;

  /* compute permutation of modes */
  fib_mode_order(tt->dims, tt->nmodes, mode, ft->dim_perm);

  /* allocate modal data */
  ft->inds = (idx_t *) malloc(ft->nnz * sizeof(idx_t));
  ft->vals = (val_t *) malloc(ft->nnz * sizeof(val_t));

  tt_sort(tt, mode, ft->dim_perm);
  if(tile != SPLATT_NOTILE) {
    ft->tiled = 1;
    tt_tile(tt, ft->dim_perm);
  }

  __create_fptr(ft, tt, mode);

  switch(ft->tiled) {
  case SPLATT_NOTILE:
    __create_sliceptr(ft, tt, mode);
    break;
  case SPLATT_COOPTILE:
    __create_slabptr(ft, tt, mode);
    break;
  }

  /* copy indmap if necessary */
  if(tt->indmap[mode] != NULL) {
    ft->indmap = (idx_t *) malloc(ft->dims[mode] * sizeof(idx_t));
    memcpy(ft->indmap, tt->indmap[mode], ft->dims[mode] * sizeof(idx_t));
  } else {
    ft->indmap = NULL;
  }
}


spmatrix_t * ften_spmat(
  ftensor_t * const ft)
{
  idx_t const nrows = ft->nfibs;
  idx_t const ncols = ft->dims[ft->dim_perm[2]];
  spmatrix_t * mat = spmat_alloc(nrows, ncols, ft->nnz);

  memcpy(mat->rowptr, ft->fptr, (nrows+1) * sizeof(idx_t));
  memcpy(mat->colind, ft->inds, ft->nnz * sizeof(idx_t));
  memcpy(mat->vals,   ft->vals, ft->nnz * sizeof(val_t));

  return mat;
}


void ften_free(
  ftensor_t * ft)
{
  free(ft->fptr);
  free(ft->fids);
  free(ft->inds);
  free(ft->vals);
  free(ft->sptr);
  free(ft->indmap);
  if(ft->tiled != SPLATT_NOTILE) {
    free(ft->slabptr);
    free(ft->sids);
  }
}


void fib_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const mode,
  idx_t * const perm_dims)
{
  perm_dims[0] = mode;
#if SPLATT_LONG_FIB == 1
  /* find largest mode */
  idx_t maxm = (mode+1) % nmodes;
  for(idx_t mo=1; mo < nmodes; ++mo) {
    if(dims[(mode+mo) % nmodes] > dims[maxm]) {
      maxm = (mode+mo) % nmodes;
    }
  }
#else
  /* find shortest mode */
  idx_t maxm = (mode+1) % nmodes;
  for(idx_t mo=1; mo < nmodes; ++mo) {
    if(dims[(mode+mo) % nmodes] < dims[maxm]) {
      maxm = (mode+mo) % nmodes;
    }
  }
#endif

  /* fill in mode permutation */
  perm_dims[nmodes-1] = maxm;
  idx_t mark = 1;
  for(idx_t mo=1; mo < nmodes; ++mo) {
    idx_t mround = (mode + mo) % nmodes;
    if(mround != maxm) {
      perm_dims[mark++] = mround;
    }
  }
}

size_t ften_storage(
  ftensor_t const * const ft)
{
  /* calculate storage */
  size_t bytes = 0;

  bytes += ft->nnz * (sizeof(idx_t) + sizeof(val_t)); /* nnz */
  bytes += (ft->nfibs + 1) * sizeof(idx_t);           /* fptr */
  bytes += ft->nfibs * sizeof(idx_t);                 /* fids */

  if(!ft->tiled != SPLATT_NOTILE) {
    bytes += (ft->nslcs + 1) * sizeof(idx_t);         /* sptr */
  } else {
    bytes += (ft->nslabs + 1) * sizeof(idx_t);        /* slabptr */
    bytes += (ft->nfibs) * sizeof(idx_t);             /* sids */
  }

  return bytes;
}

