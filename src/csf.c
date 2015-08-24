
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
  ctensor_t const * const ct)
{
  printf("-----------\n");
  printf("nmodes: %lu nnz: %lu ntiles: %lu\n", ct->nmodes, ct->nnz, ct->ntiles);
  printf("dims: %lu", ct->dims[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("x%lu", ct->dims[m]);
  }
  printf(" (%lu", ct->dim_perm[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("->%lu", ct->dim_perm[m]);
  }
  printf(")\n");

  for(idx_t t=0; t < ct->ntiles; ++t) {
    csf_sparsity_t const * const ft = ct->pt + t;
    /* skip empty tiles */
    if(ft->vals == NULL) {
      continue;
    }

    /* write slices */
    printf("tile: %lu fptr:\n", t);
    printf("[%lu] ", ft->nfibs[0]);
    for(idx_t f=0; f < ft->nfibs[0]; ++f) {
      if(ft->fids[0] == NULL) {
        printf(" %lu", ft->fptr[0][f]);
      } else {
        printf(" (%lu, %lu)", ft->fptr[0][f], ft->fids[0][f]);
      }
    }
    printf(" %lu\n", ft->fptr[0][ft->nfibs[0]]);

    /* inner nodes */
    for(idx_t m=1; m < ct->nmodes-1; ++m) {
      printf("[%lu] ", ft->nfibs[m]);
      for(idx_t f=0; f < ft->nfibs[m]; ++f) {
        printf(" (%lu, %lu)", ft->fptr[m][f], ft->fids[m][f]);
      }
      printf(" %lu\n", ft->fptr[m][ft->nfibs[m]]);
    }

    /* vals/inds */
    printf("[%lu] ", ft->nfibs[ct->nmodes-1]);
    for(idx_t f=0; f < ft->nfibs[ct->nmodes-1]; ++f) {
      printf(" %3lu", ft->fids[ct->nmodes-1][f]);
    }
    printf("\n");
    for(idx_t n=0; n < ft->nfibs[ct->nmodes-1]; ++n) {
      printf(" %0.1f", ft->vals[n]);
    }
    printf("\n");
  }

  printf("-----------\n\n");
}


static void __mk_outerptr(
  ctensor_t * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr)
{
  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  assert(nnzstart < nnzend);

  /* the mode after accounting for dim_perm */
  idx_t const * const restrict ttind = tt->ind[ct->dim_perm[0]] + nnzstart;

  /* count fibers */
  idx_t nfibs = 1;
  for(idx_t x=1; x < nnz; ++x) {
    assert(ttind[x-1] <= ttind[x]);
    if(ttind[x] != ttind[x-1]) {
      ++nfibs;
    }
  }
  ct->pt[tile_id].nfibs[0] = nfibs;
  assert(nfibs <= ct->dims[ct->dim_perm[0]]);

  /* grab sparsity pattern */
  csf_sparsity_t * const pt = ct->pt + tile_id;

  pt->fptr[0] = (idx_t *) malloc((nfibs+1) * sizeof(idx_t));
  if(ct->tile_dims[ct->dim_perm[0]] > 1) {
    pt->fids[0] = (idx_t *) malloc(nfibs * sizeof(idx_t));
  } else {
    pt->fids[0] = NULL;
  }

  idx_t  * const restrict fp = pt->fptr[0];
  idx_t  * const restrict fi = pt->fids[0];
  fp[0] = 0;
  if(fi != NULL) {
    fi[0] = ttind[0];
  }

  idx_t nfound = 1;
  for(idx_t n=1; n < nnz; ++n) {
    /* check for end of outer index */
    if(ttind[n] != ttind[n-1]) {
      if(fi != NULL) {
        fi[nfound] = ttind[n];
      }
      fp[nfound++] = n;
    }
  }

  fp[nfibs] = nnz;
}


static void __mk_fptr(
  ctensor_t * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr,
  idx_t const mode)
{
  assert(mode < ct->nmodes);

  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  /* outer mode is easy; just look at outer indices */
  if(mode == 0) {
    __mk_outerptr(ct, tt, tile_id, nnztile_ptr);
    return;
  }
  /* the mode after accounting for dim_perm */
  idx_t const * const restrict ttind = tt->ind[ct->dim_perm[mode]] + nnzstart;

  csf_sparsity_t * const pt = ct->pt + tile_id;

  /* we will edit this to point to the new fiber idxs instead of nnz */
  idx_t * const restrict fprev = pt->fptr[mode-1];

  /* first count nfibers */
  idx_t nfibs = 0;
  /* foreach 'slice' in the previous dimension */
  for(idx_t s=0; s < pt->nfibs[mode-1]; ++s) {
    ++nfibs; /* one by default per 'slice' */
    /* count fibers in current hyperplane*/
    for(idx_t f=fprev[s]+1; f < fprev[s+1]; ++f) {
      if(ttind[f] != ttind[f-1]) {
        ++nfibs;
      }
    }
  }
  pt->nfibs[mode] = nfibs;


  pt->fptr[mode] = (idx_t *) malloc((nfibs+1) * sizeof(idx_t));
  pt->fids[mode] = (idx_t *) malloc(nfibs * sizeof(idx_t));
  idx_t * const restrict fp = pt->fptr[mode];
  idx_t * const restrict fi = pt->fids[mode];
  fp[0] = 0;

  /* now fill in fiber info */
  idx_t nfound = 0;
  for(idx_t s=0; s < pt->nfibs[mode-1]; ++s) {
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
  fprev[pt->nfibs[mode-1]] = nfibs;
  fp[nfibs] = nnz;
}


/**
* @brief Allocate and fill a CSF tensor from a coordinate tensor without
*        tiling.
*
* @param ft The CSF tensor to fill out.
* @param tt The sparse tensor to start from.
*/
static void __ctensor_alloc_untiled(
  ctensor_t * const ct,
  sptensor_t * const tt)
{
  idx_t const nmodes = tt->nmodes;
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);

  ct->ntiles = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ct->tile_dims[m] = 1;
  }
  ct->pt = (csf_sparsity_t *) malloc(sizeof(csf_sparsity_t));

  csf_sparsity_t * const pt = ct->pt;

  /* last row of fptr is just nonzero inds */
  pt->nfibs[nmodes-1] = ct->nnz;
  pt->fids[nmodes-1] = (idx_t *) malloc(ct->nnz * sizeof(idx_t));
  pt->vals           = (val_t *) malloc(ct->nnz * sizeof(val_t));
  memcpy(pt->fids[nmodes-1], tt->ind[ct->dim_perm[nmodes-1]],
      ct->nnz * sizeof(idx_t));
  memcpy(pt->vals, tt->vals, ct->nnz * sizeof(val_t));

  /* setup a basic tile ptr for one tile */
  idx_t nnz_ptr[2];
  nnz_ptr[0] = 0;
  nnz_ptr[1] = tt->nnz;

  /* create fptr entries for the rest of the modes, working down from roots.
   * Skip the bottom level (nnz) */
  for(idx_t m=0; m < tt->nmodes-1; ++m) {
    __mk_fptr(ct, tt, 0, nnz_ptr, m);
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
  ctensor_t * const ct,
  sptensor_t * const tt,
  double const * const splatt_opts)
{
  idx_t const nmodes = tt->nmodes;

  idx_t ntiles = 1;
  for(idx_t m=0; m < ct->nmodes; ++m) {
    ct->tile_dims[m] = (idx_t) splatt_opts[SPLATT_OPTION_NTHREADS];
    ntiles *= ct->tile_dims[m];
  }
  /* perform tensor tiling */
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);
  idx_t * nnz_ptr = tt_densetile(tt, ct->tile_dims);

  ct->ntiles = ntiles;
  ct->pt = (csf_sparsity_t *) malloc(ntiles * sizeof(csf_sparsity_t));

  for(idx_t t=0; t < ntiles; ++t) {
    idx_t const startnnz = nnz_ptr[t];
    idx_t const endnnz   = nnz_ptr[t+1];
    idx_t const ptnnz = endnnz - startnnz;

    csf_sparsity_t * const pt = ct->pt + t;

    /* empty tile */
    if(ptnnz == 0) {
      for(idx_t m=0; m < ct->nmodes; ++m) {
        pt->fptr[m] = NULL;
        pt->fids[m] = NULL;
        pt->nfibs[m] = 0;
      }
      /* first fptr may be accessed anyway */
      pt->fptr[0] = (idx_t *) malloc(2 * sizeof(idx_t));
      pt->fptr[0][0] = 0;
      pt->fptr[0][1] = 0;
      pt->vals = NULL;
      continue;
    }

    /* last row of fptr is just nonzero inds */
    pt->nfibs[nmodes-1] = ptnnz;

    pt->fids[nmodes-1] = (idx_t *) malloc(ptnnz * sizeof(idx_t));
    memcpy(pt->fids[nmodes-1], tt->ind[ct->dim_perm[nmodes-1]] + startnnz,
        ptnnz * sizeof(idx_t));

    pt->vals = (val_t *) malloc(ptnnz * sizeof(val_t));
    memcpy(pt->vals, tt->vals + startnnz, ptnnz * sizeof(val_t));

    /* create fptr entries for the rest of the modes*/
    for(idx_t m=0; m < tt->nmodes-1; ++m) {
      __mk_fptr(ct, tt, t, nnz_ptr, m);
    }
  }

  free(nnz_ptr);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void ctensor_alloc(
  ctensor_t * const ct,
  sptensor_t * const tt,
  double const * const splatt_opts)
{
  ct->nnz = tt->nnz;
  ct->nmodes = tt->nmodes;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ct->dims[m] = tt->dims[m];
  }

  /* get the indices in order */
  csf_find_mode_order(tt->dims, tt->nmodes, CSF_SORTED_SMALLFIRST,
      ct->dim_perm);

  splatt_tile_t which_tile = (splatt_tile_t) splatt_opts[SPLATT_OPTION_TILE];
  switch(which_tile) {
  case SPLATT_NOTILE:
    __ctensor_alloc_untiled(ct, tt);
    break;
  case SPLATT_DENSETILE:
    __csf_alloc_densetile(ct, tt, splatt_opts);
    break;
  default:
    fprintf(stderr, "SPLATT: tiling '%d' unsupported for CSF tensors.\n",
        which_tile);
    break;
  }
#if 0
  if(tt->nnz < 100) {
    tt_write(tt, NULL);
    __print_csf(ct);
  }
#endif
}



void ctensor_free(
  ctensor_t * const ct)
{
  /* free each tile of sparsity pattern */
  for(idx_t t=0; t < ct->ntiles; ++t) {
    free(ct->pt[t].vals);
    free(ct->pt[t].fids[ct->nmodes-1]);
    for(idx_t m=0; m < ct->nmodes-1; ++m) {
      free(ct->pt[t].fptr[m]);
      free(ct->pt[t].fids[m]);
    }
  }

  free(ct->pt);
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


idx_t ctensor_storage(
  ctensor_t const * const ct)
{
  idx_t bytes = 0;
  bytes += ct->nnz * sizeof(val_t); /* vals */
  bytes += ct->nnz * sizeof(idx_t); /* fids[nmodes] */
  for(idx_t t=0; t < ct->ntiles; ++t) {
    csf_sparsity_t const * const pt = ct->pt + t;

    for(idx_t m=0; m < ct->nmodes-1; ++m) {
      bytes += (pt->nfibs[m]+1) * sizeof(idx_t); /* fptr */
      if(pt->fids[m] != NULL) {
        bytes += pt->nfibs[m] * sizeof(idx_t); /* fids */
      }
    }
  }
  return bytes;
}


