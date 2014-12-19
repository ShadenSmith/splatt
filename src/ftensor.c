

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sort.h"
#include "io.h"
#include "matrix.h"
#include "ftensor.h"
#include "tile.h"


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
    ttinds[m] = tt->ind[ft->dim_perms[mode][m]];
  }
  /* this avoids some maybe-uninitialized warnings */
  for(idx_t m=nmodes; m < MAX_NMODES; ++m) {
    ttinds[m] = NULL;
  }

  /* count fibers and copy inds/vals into ft */
  idx_t nfibs = 1;
  ft->inds[mode][0] = ttinds[nmodes-1][0];
  ft->vals[mode][0] = tt->vals[0];

  /* count fibers in tt */
  for(idx_t n=1; n < nnz; ++n) {
    for(idx_t m=0; m < nmodes-1; ++m) {
      /* check for new fiber */
      if(ttinds[m][n] != ttinds[m][n-1]) {
        ++nfibs;
        break;
      }
    }
    ft->inds[mode][n] = ttinds[nmodes-1][n];
    ft->vals[mode][n] = tt->vals[n];
  }

  printf("mode: "SS_IDX" found "SS_IDX" fibers.\n", mode, nfibs);

  /* allocate fiber structure */
  ft->nfibs[mode] = nfibs;
  ft->fptr[mode] = (idx_t *) malloc((nfibs+1) * sizeof(idx_t));
  ft->fids[mode] = (idx_t *) malloc(nfibs * sizeof(idx_t));
  if(ft->tiled) {
    ft->sids[mode]= (idx_t *) malloc(nfibs * sizeof(idx_t));
  }

  /* initialize boundary values */
  ft->fptr[mode][0] = 0;
  ft->fptr[mode][nfibs] = nnz;
  ft->fids[mode][0] = ttinds[1][0];
  if(ft->tiled) {
    ft->sids[mode][0] = ttinds[0][0];
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
      ft->fptr[mode][fib] = n;
      ft->fids[mode][fib] = ttinds[1][n];
      if(ft->tiled) {
        ft->sids[mode][fib] = ttinds[0][n];
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

  ft->nslabs[mode] = nslabs;
  ft->slabptr[mode] = (idx_t *) malloc((nslabs+1) * sizeof(idx_t));

  idx_t slab = 0;
  ft->slabptr[mode][slab++] = 0;

  idx_t const nfibs = ft->nfibs[mode];
  for(idx_t f=0; f < nfibs; ++f) {
    idx_t const slice = ft->sids[mode][f];
    /* update slabptr as necessary and account for empty slabs (hopefully
     * unlikely...*/
    while(slice / tsize > slab-1) {
      ft->slabptr[mode][slab++] = f;
    }
  }
  ft->slabptr[mode][slab] = nfibs;
  assert(slab == nslabs);
}


static void __create_sliceptr(
  ftensor_t * const ft,
  sptensor_t const * const tt,
  idx_t const mode)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;

  idx_t const nslices = ft->dims[mode];
  ft->sptr[mode] = (idx_t *) malloc((nslices+1) * sizeof(idx_t));

  /* permuted tt->ind makes things a bit easier */
  idx_t * ttinds[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    ttinds[m] = tt->ind[ft->dim_perms[mode][m]];
  }
  /* this avoids some maybe-uninitialized warnings */
  for(idx_t m=nmodes; m < MAX_NMODES; ++m) {
    ttinds[m] = NULL;
  }

  idx_t slice = 0;
  ft->sptr[mode][slice++] = 0;
  while(slice != ttinds[0][0]+1) {
    ft->sptr[mode][slice++] = 0;
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
        ft->sptr[mode][slice++] = fib;
      }
      ++fib;
    }
  }
  /* account for any empty slices at end */
  for(idx_t s=slice; s <= ft->dims[mode]; ++s) {
    ft->sptr[mode][s] = ft->nfibs[mode];
  }
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
ftensor_t * ften_alloc(
  sptensor_t * const tt,
  int const tile)
{
  ftensor_t * ft = (ftensor_t *) malloc(sizeof(ftensor_t));
  ft->nnz = tt->nnz;
  ft->nmodes = tt->nmodes;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ft->dims[m] = tt->dims[m];
  }

  ft->tiled = tt->tiled;

  /* compute permutation of modes */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    fib_mode_order(tt->dims, tt->nmodes, m, ft->dim_perms[m]);
  }

  /* allocate modal data */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    ft->inds[m] = (idx_t *) malloc(ft->nnz * sizeof(idx_t));
    ft->vals[m] = (val_t *) malloc(ft->nnz * sizeof(val_t));

    tt_sort(tt, m, ft->dim_perms[m]);
    if(tile) {
      tt_tile(tt, ft->dim_perms[m]);
      ft->tiled = 1;
    }

    __create_fptr(ft, tt, m);
    if(ft->tiled) {
      __create_slabptr(ft, tt, m);
    } else {
      __create_sliceptr(ft, tt, m);
    }
  }

  return ft;
}


spmatrix_t * ften_spmat(
  ftensor_t * ft,
  idx_t const mode)
{
  idx_t const nrows = ft->nfibs[mode];
  idx_t const ncols = ft->dims[ft->dim_perms[mode][2]];
  spmatrix_t * mat = spmat_alloc(nrows, ncols, ft->nnz);

  memcpy(mat->rowptr, ft->fptr[mode], (nrows+1) * sizeof(idx_t));
  memcpy(mat->colind, ft->inds[mode], ft->nnz * sizeof(idx_t));
  memcpy(mat->vals,   ft->vals[mode], ft->nnz * sizeof(val_t));

  return mat;
}


void ften_free(
  ftensor_t * ft)
{
  for(idx_t m=0; m < ft->nmodes; ++m) {
    free(ft->fptr[m]);
    free(ft->fids[m]);
    free(ft->inds[m]);
    free(ft->vals[m]);
    if(ft->tiled) {
      free(ft->slabptr[m]);
      free(ft->sids[m]);
    } else {
      free(ft->sptr[m]);
    }
  }
  free(ft);
}


void fib_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const mode,
  idx_t * const perm_dims)
{
  perm_dims[0] = mode;
  /* find largest mode */
  idx_t maxm = (mode+1) % nmodes;
  for(idx_t mo=1; mo < nmodes; ++mo) {
    if(dims[(mode+mo) % nmodes] > dims[maxm]) {
      maxm = (mode+mo) % nmodes;
    }
  }

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

