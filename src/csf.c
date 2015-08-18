
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "csf.h"
#include "sort.h"
#include "tile.h"

#include "io.h"

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static void __order_dims_small(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t * const perm_dims)
{
  idx_t sorted[MAX_NMODES];
  idx_t matched[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    sorted[m] = dims[m];
    matched[m] = 0;
  }
  quicksort(sorted, nmodes);

  /* silly n^2 comparison to grab modes from sorted dimensions.
   * TODO: make a key/val sort...*/
  for(idx_t mfind=0; mfind < nmodes; ++mfind) {
    for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
      if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
        perm_dims[mfind] = mcheck;
        matched[mcheck] = 1;
        break;
      }
    }
  }
}


static void __order_dims_large(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t * const perm_dims)
{
  idx_t sorted[MAX_NMODES];
  idx_t matched[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    sorted[m] = dims[m];
    matched[m] = 0;
  }
  /* sort small -> large */
  quicksort(sorted, nmodes);

  /* reverse list */
  for(idx_t m=0; m < nmodes/2; ++m) {
    idx_t tmp = sorted[nmodes-m-1];
    sorted[nmodes-m-1] = sorted[m];
    sorted[m] = tmp;
  }

  /* silly n^2 comparison to grab modes from sorted dimensions.
   * TODO: make a key/val sort...*/
  for(idx_t mfind=0; mfind < nmodes; ++mfind) {
    for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
      if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
        perm_dims[mfind] = mcheck;
        matched[mcheck] = 1;
        break;
      }
    }
  }

}


static void __print_csf(
  csf_t const * const ft)
{
  printf("-----------\n");
  printf("nmodes: %lu nnz: %lu\n", ft->nmodes, ft->nnz);
  printf("dims: %lu", ft->dims[0]);
  for(idx_t m=1; m < ft->nmodes; ++m) {
    printf("x%lu", ft->dims[m]);
  }
  printf(" (%lu", ft->dim_perm[0]);
  for(idx_t m=1; m < ft->nmodes; ++m) {
    printf("->%lu", ft->dim_perm[m]);
  }
  printf(")\n");

  /* write slices */
  printf("fptr:\n");
  printf("[%lu] ", ft->nfibs[0]);
  for(idx_t f=0; f < ft->nfibs[0]; ++f) {
    printf(" %lu", ft->fptr[0][f]);
  }
  printf(" %lu\n", ft->fptr[0][ft->nfibs[0]]);

  /* inner nodes */
  for(idx_t m=1; m < ft->nmodes-1; ++m) {
    printf("[%lu] ", ft->nfibs[m]);
    for(idx_t f=0; f < ft->nfibs[m]; ++f) {
      printf(" (%lu, %lu)", ft->fptr[m][f], ft->fids[m][f]);
    }
    printf(" %lu\n", ft->fptr[m][ft->nfibs[m]]);
  }

  /* vals/inds */
  printf("[%lu] ", ft->nfibs[ft->nmodes-1]);
  for(idx_t f=0; f < ft->nfibs[ft->nmodes-1]; ++f) {
    printf(" %3lu", ft->fids[ft->nmodes-1][f]);
  }
  printf("\n");
  for(idx_t n=0; n < ft->nnz; ++n) {
    printf(" %0.1f", ft->vals[n]);
  }
  printf("\n");

  printf("-----------\n\n");
}


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


static void __mk_tiled_outerptr(
  csf_t * const ft,
  sptensor_t const * const tt,
  idx_t const * const tile_ptr,
  idx_t const mode)
{
  /* the mode after accounting for dim_perm */
  idx_t const mperm = ft->dim_perm[mode];
  idx_t const * const restrict ttind = tt->ind[mperm];

  /* Count fibers summed across all tiles. Each tile will need its own
   * start/end fibers */
  idx_t nfibs = 0;
  for(idx_t i=0; i < ft->tile_dims[mode]; ++i) {
    idx_t id;
    id = get_next_tileid(TILE_BEGIN, ft->tile_dims, tt->nmodes, 0, i);
    while(id != TILE_END) {
      idx_t const start = tile_ptr[id];
      idx_t const end = tile_ptr[id+1];
      for(idx_t m=0; m < tt->nmodes; ++m) {
        for(idx_t x=start; x < end; ++x) {
        }
      }

      /* next tile */
      id = get_next_tileid(id, ft->tile_dims, tt->nmodes, 0, i);
    }
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


/**
* @brief Allocate and fill a CSF tensor from a coordinate tensor without
*        tiling.
*
* @param ft The CSF tensor to fill out.
* @param tt The sparse tensor to start from.
*/
static void __csf_alloc_untiled(
  csf_t * const ft,
  sptensor_t * const tt)
{
  idx_t const nmodes = tt->nmodes;
  tt_sort(tt, ft->dim_perm[0], ft->dim_perm);

  /* last row of fptr is just nonzero inds */
  ft->nfibs[nmodes-1] = ft->nnz;
  ft->fids[nmodes-1] = (idx_t *) malloc(ft->nnz * sizeof(idx_t));
  ft->vals           = (val_t *) malloc(ft->nnz * sizeof(val_t));
  memcpy(ft->fids[nmodes-1], tt->ind[ft->dim_perm[nmodes-1]],
      ft->nnz * sizeof(idx_t));
  memcpy(ft->vals, tt->vals, ft->nnz * sizeof(val_t));

  /* create fptr entries for the rest of the modes, working up from */
  for(idx_t m=0; m < tt->nmodes-1; ++m) {
    __mk_fptr(ft, tt, m);
  }
}


/**
* @brief Reorder the nonzeros in a sparse tensor using dense tiling and fill
*        a CSF tensor with the data.
*
* @param ft The CSF tensor to fill.
* @param tt The sparse tensor to start from.
* @param splatt_opts Options array for SPLATT - used for tile dimensions.
*/
static void __csf_alloc_densetile(
  csf_t * const ft,
  sptensor_t * const tt,
  double const * const splatt_opts)
{
  idx_t const nmodes = tt->nmodes;

  idx_t ntiles = 1;
  for(idx_t m=0; m < ft->nmodes; ++m) {
    ft->tile_dims[m] = (idx_t) splatt_opts[SPLATT_OPTION_NTHREADS];
    ntiles *= ft->tile_dims[m];
  }
  idx_t * nnz_ptr = tt_densetile(tt, ft->tile_dims);

  /* last row of fptr is just nonzero inds */
  ft->nfibs[nmodes-1] = ft->nnz;
  ft->fids[nmodes-1] = (idx_t *) malloc(ft->nnz * sizeof(idx_t));
  ft->vals           = (val_t *) malloc(ft->nnz * sizeof(val_t));
  memcpy(ft->fids[nmodes-1], tt->ind[ft->dim_perm[nmodes-1]],
      ft->nnz * sizeof(idx_t));
  memcpy(ft->vals, tt->vals, ft->nnz * sizeof(val_t));

  /* XXX: do we account for dim_perm instead of 0 here? */
  __mk_tiled_outerptr(ft, tt, nnz_ptr, 0);
  /* create fptr entries for the rest of the modes, working up from */
  for(idx_t m=0; m < tt->nmodes-1; ++m) {
    //__mk_fptr(ft, tt, m);
  }

  free(nnz_ptr);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void csf_alloc(
  csf_t * const ft,
  sptensor_t * const tt,
  double const * const splatt_opts)
{
  ft->nmodes = tt->nmodes;
  ft->nnz = tt->nnz;
  ft->fptr = (idx_t **) malloc(tt->nmodes * sizeof(idx_t *));
  ft->fids = (idx_t **) malloc(tt->nmodes * sizeof(idx_t *));

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ft->dims[m] = tt->dims[m];
    ft->fptr[m] = NULL;
    ft->fids[m] = NULL;
  }

  /* get the indices in order */
  csf_find_mode_order(tt->dims, tt->nmodes, CSF_SORTED_SMALLFIRST,
      ft->dim_perm);

  splatt_tile_t which_tile = (splatt_tile_t) splatt_opts[SPLATT_OPTION_TILE];
  switch(which_tile) {
  case SPLATT_NOTILE:
    __csf_alloc_untiled(ft, tt);
    break;
  case SPLATT_DENSETILE:
    __csf_alloc_densetile(ft, tt, splatt_opts);
    break;
  default:
    fprintf(stderr, "SPLATT: tiling '%d' unsupported for CSF tensors.\n",
        which_tile);
    break;
  }
#if 0
  tt_write(tt, NULL);
  __print_csf(ft);
#endif
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



void csf_find_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  csf_mode_type which,
  idx_t * const perm_dims)
{
  switch(which) {
  case CSF_SORTED_SMALLFIRST:
    __order_dims_small(dims, nmodes, perm_dims);
    break;
  case CSF_SORTED_BIGFIRST:
    fprintf(stderr, "SPLATT: using 'CSF_SORTED_BIGFIRST' for csf_alloc. "
                    "Not recommended.\n");
    __order_dims_large(dims, nmodes, perm_dims);
    break;
  default:
    fprintf(stderr, "SPLATT: csf_mode_type '%d' not recognized.\n", which);
    break;
  }
}


idx_t csf_storage(
  csf_t const * const ft)
{
  idx_t bytes = 0;
  bytes += ft->nnz * sizeof(val_t); /* vals */
  bytes += ft->nnz * sizeof(idx_t); /* fids[nmodes] */
  for(idx_t m=0; m < ft->nmodes-1; ++m) {
    bytes += (ft->nfibs[m]+1) * sizeof(idx_t); /* fptr */
    /* only look at fids for non-outer mode */
    if(m > 0) {
      bytes += ft->nfibs[m] * sizeof(idx_t); /* fids */
    }
  }
  return bytes;
}


