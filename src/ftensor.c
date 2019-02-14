

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
static void p_create_fptr(
  ftensor_t * const ft,
  splatt_coo const * const tt,
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
  ft->fptr = (idx_t *) splatt_malloc((nfibs+1) * sizeof(idx_t));
  ft->fids = (idx_t *) splatt_malloc(nfibs * sizeof(idx_t));
  if(ft->tiled != SPLATT_NOTILE) {
    /* temporary and will be replaced later */
    ft->sids = (idx_t *) splatt_malloc(nfibs * sizeof(idx_t));
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


static void p_create_syncptr(
  ftensor_t * const ft,
  splatt_coo const * const tt,
  idx_t const mode)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;
  idx_t const tsize = TILE_SIZES[0];
  idx_t const nslabs = tt->dims[mode] / tsize + (tt->dims[mode] % tsize != 0);
  idx_t const nfibs = ft->nfibs;

  ft->sptr = NULL; /* not needed */
  ft->slabptr = (idx_t *) splatt_malloc((nslabs+1) * sizeof(idx_t));

  ft->slabptr[0] = 0;
  idx_t slab = 1;
  for(idx_t f=1; f < nfibs; ++f) {
    /* update slabptr if we've moved to the next slab */
    if(ft->sids[f] / tsize != slab-1) {
      ft->slabptr[slab++] = f;
    }
  }

  ft->nslabs = slab;
  ft->slabptr[slab] = nfibs;
}



static void p_create_slabptr(
  ftensor_t * const ft,
  splatt_coo const * const tt,
  idx_t const mode)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;
  idx_t const tsize = TILE_SIZES[0];
  idx_t const nslabs = tt->dims[mode] / tsize + (tt->dims[mode] % tsize != 0);

  ft->slabptr = (idx_t *) splatt_malloc((nslabs+1) * sizeof(idx_t));

  idx_t const nfibs = ft->nfibs;
  /* count slices */
  idx_t slices = 1;
  for(idx_t f=1; f < nfibs; ++f) {
    if(ft->sids[f] != ft->sids[f-1]) {
      ++slices;
    }
  }
  idx_t * sptr = (idx_t *) splatt_malloc((slices+1) * sizeof(idx_t)); /* sliceptr */
  idx_t * sids = (idx_t *) splatt_malloc(slices * sizeof(idx_t)); /* to replace old */

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

  /* update ft with new data structures */
  free(ft->sids);
  ft->sids = sids;
  ft->sptr = sptr;

  /* account for any empty slabs at end */
  ft->nslabs = slab;
  ft->slabptr[slab] = slices;
  ft->sptr[slices] = nfibs;
}


static void p_create_sliceptr(
  ftensor_t * const ft,
  splatt_coo const * const tt,
  idx_t const mode)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;

  idx_t const nslices = ft->dims[mode];
  ft->sptr = (idx_t *) splatt_malloc((nslices+1) * sizeof(idx_t));
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
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void ften_alloc(
  ftensor_t * const ft,
  splatt_coo * const tt,
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
  ft->inds = (idx_t *) splatt_malloc(ft->nnz * sizeof(idx_t));
  ft->vals = (val_t *) splatt_malloc(ft->nnz * sizeof(val_t));

  tt_sort(tt, mode, ft->dim_perm);
  if(tile != SPLATT_NOTILE) {
    ft->tiled = tile;
    tt_tile(tt, ft->dim_perm);
  }

  p_create_fptr(ft, tt, mode);

  switch(ft->tiled) {
  case SPLATT_NOTILE:
    p_create_sliceptr(ft, tt, mode);
    break;
  case SPLATT_SYNCTILE:
    p_create_syncptr(ft, tt, mode);
    break;
  case SPLATT_COOPTILE:
    p_create_slabptr(ft, tt, mode);
    break;
  default:
    fprintf(stderr, "SPLATT: tile type '%d' not recognized.\n", ft->tiled);
    /* just default to no tiling */
    p_create_sliceptr(ft, tt, mode);
    ft->tiled = SPLATT_NOTILE;
  }

  /* copy indmap if necessary */
  if(tt->indmap[mode] != NULL) {
    ft->indmap = (idx_t *) splatt_malloc(ft->dims[mode] * sizeof(idx_t));
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

  switch(ft->tiled) {
  case SPLATT_SYNCTILE:
    free(ft->slabptr);
    free(ft->sids);
    break;

  case SPLATT_COOPTILE:
    free(ft->slabptr);
    free(ft->sids);
    break;
  default:
    break;
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

