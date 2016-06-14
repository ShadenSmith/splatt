
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "csf.h"
#include "sort.h"
#include "tile.h"

#include "io.h"


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

int splatt_csf_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_csf ** tensors,
    double const * const options)
{
  sptensor_t * tt = tt_read(fname);
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  tt_remove_empty(tt);

  *tensors = csf_alloc(tt, options);
  *nmodes = tt->nmodes;

  tt_free(tt);

  return SPLATT_SUCCESS;
}

int splatt_csf_convert(
    splatt_idx_t const nmodes,
    splatt_idx_t const nnz,
    splatt_idx_t ** const inds,
    splatt_val_t * const vals,
    splatt_csf ** tensors,
    double const * const options)
{
  sptensor_t tt;
  tt_fill(&tt, nnz, nmodes, inds, vals);
  tt_remove_empty(&tt);

  *tensors = csf_alloc(&tt, options);

  return SPLATT_SUCCESS;
}


void splatt_free_csf(
    splatt_csf * tensors,
    double const * const options)
{
  csf_free(tensors, options);
}




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Find a permutation of modes that results in non-increasing mode size.
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_small(
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

/**
* @brief Find a permutation of modes such that the first mode is 'custom-mode'
*        and the remaining are naturally ordered (0, 1, ...).
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param custom_mode The mode to place first.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_inorder(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const custom_mode,
  idx_t * const perm_dims)
{
  /* initialize to natural ordering */
  for(idx_t m=0; m < nmodes; ++m) {
    perm_dims[m] = m;
  }

  /* find where custom_mode was placed and adjust from there */
  for(idx_t m=0; m < nmodes; ++m) {
    if(perm_dims[m] == custom_mode) {
      memmove(perm_dims + 1, perm_dims, (m) * sizeof(m));
      perm_dims[0] = custom_mode;
      break;
    }
  }
}


/**
* @brief Find a permutation of modes such that the first mode is 'custom-mode'
*        and the remaining are sorted in non-increasing order.
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param custom_mode The mode to place first.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_minusone(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const custom_mode,
  idx_t * const perm_dims)
{
  p_order_dims_small(dims, nmodes, perm_dims);

  /* find where custom_mode was placed and adjust from there */
  for(idx_t m=0; m < nmodes; ++m) {
    if(perm_dims[m] == custom_mode) {
      memmove(perm_dims + 1, perm_dims, (m) * sizeof(m));
      perm_dims[0] = custom_mode;
      break;
    }
  }
}


/**
* @brief Find a permutation of modes that results in non-decreasing mode size.
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_large(
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


/**
* @brief Construct the sparsity structure of the outer-mode of a CSF tensor.
*
* @param ct The CSF tensor to construct.
* @param tt The coordinate tensor to construct from. Assumed to be already
*            sorted.
* @param tile_id The ID of the tile to construct.
* @param nnztile_ptr A pointer into 'tt' that marks the start of each tile.
*/
static void p_mk_outerptr(
  splatt_csf * const ct,
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
  csf_sparsity * const pt = ct->pt + tile_id;

  pt->fptr[0] = splatt_malloc((nfibs+1) * sizeof(**(pt->fptr)));
  if(ct->ntiles > 1) {
    pt->fids[0] = splatt_malloc(nfibs * sizeof(**(pt->fids)));
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


/**
* @brief Construct the sparsity structure of any mode but the last. The first
*        (root) mode is handled by p_mk_outerptr and the first is simply a copy
*        of the nonzeros.
*
* @param ct The CSF tensor to construct.
* @param tt The coordinate tensor to construct from. Assumed to be already
*            sorted.
* @param tile_id The ID of the tile to construct.
* @param nnztile_ptr A pointer into 'tt' that marks the start of each tile.
* @param mode Which mode we are constructing.
*/
static void p_mk_fptr(
  splatt_csf * const ct,
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
    p_mk_outerptr(ct, tt, tile_id, nnztile_ptr);
    return;
  }
  /* the mode after accounting for dim_perm */
  idx_t const * const restrict ttind = tt->ind[ct->dim_perm[mode]] + nnzstart;

  csf_sparsity * const pt = ct->pt + tile_id;

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


  pt->fptr[mode] = splatt_malloc((nfibs+1) * sizeof(**(pt->fptr)));
  pt->fids[mode] = splatt_malloc(nfibs * sizeof(**(pt->fids)));
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
* @param ct The CSF tensor to fill out.
* @param tt The sparse tensor to start from.
*/
static void p_csf_alloc_untiled(
  splatt_csf * const ct,
  sptensor_t * const tt)
{
  idx_t const nmodes = tt->nmodes;
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);

  ct->ntiles = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ct->tile_dims[m] = 1;
  }
  ct->pt = splatt_malloc(sizeof(*(ct->pt)));

  csf_sparsity * const pt = ct->pt;

  /* last row of fptr is just nonzero inds */
  pt->nfibs[nmodes-1] = ct->nnz;
  pt->fids[nmodes-1] = splatt_malloc(ct->nnz * sizeof(**(pt->fids)));
  pt->vals           = splatt_malloc(ct->nnz * sizeof(*(pt->vals)));
  memcpy(pt->fids[nmodes-1], tt->ind[ct->dim_perm[nmodes-1]],
      ct->nnz * sizeof(**(pt->fids)));
  memcpy(pt->vals, tt->vals, ct->nnz * sizeof(*(pt->vals)));

  /* setup a basic tile ptr for one tile */
  idx_t nnz_ptr[2];
  nnz_ptr[0] = 0;
  nnz_ptr[1] = tt->nnz;

  /* create fptr entries for the rest of the modes, working down from roots.
   * Skip the bottom level (nnz) */
  for(idx_t m=0; m < tt->nmodes-1; ++m) {
    p_mk_fptr(ct, tt, 0, nnz_ptr, m);
  }
}


/**
* @brief Reorder the nonzeros in a sparse tensor using dense tiling and fill
*        a CSF tensor with the data.
*
* @param ct The CSF tensor to fill.
* @param tt The sparse tensor to start from.
* @param splatt_opts Options array for SPLATT - used for tile dimensions.
*/
static void p_csf_alloc_densetile(
  splatt_csf * const ct,
  sptensor_t * const tt,
  double const * const splatt_opts)
{
  idx_t const nmodes = tt->nmodes;

  idx_t ntiles = 1;
  for(idx_t m=0; m < ct->nmodes; ++m) {
    idx_t const depth = csf_mode_depth(m, ct->dim_perm, ct->nmodes);
    if(depth >= splatt_opts[SPLATT_OPTION_TILEDEPTH]) {
      ct->tile_dims[m] = (idx_t) splatt_opts[SPLATT_OPTION_NTHREADS];
    } else {
      ct->tile_dims[m] = 1;
    }
    ntiles *= ct->tile_dims[m];
  }

  /* perform tensor tiling */
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);
  idx_t * nnz_ptr = tt_densetile(tt, ct->tile_dims);

  ct->ntiles = ntiles;
  ct->pt = splatt_malloc(ntiles * sizeof(*(ct->pt)));

  for(idx_t t=0; t < ntiles; ++t) {
    idx_t const startnnz = nnz_ptr[t];
    idx_t const endnnz   = nnz_ptr[t+1];
    idx_t const ptnnz = endnnz - startnnz;

    csf_sparsity * const pt = ct->pt + t;

    /* empty tile */
    if(ptnnz == 0) {
      for(idx_t m=0; m < ct->nmodes; ++m) {
        pt->fptr[m] = NULL;
        pt->fids[m] = NULL;
        pt->nfibs[m] = 0;
      }
      /* first fptr may be accessed anyway */
      pt->fptr[0] = (idx_t *) splatt_malloc(2 * sizeof(**(pt->fptr)));
      pt->fptr[0][0] = 0;
      pt->fptr[0][1] = 0;
      pt->vals = NULL;
      continue;
    }

    /* last row of fptr is just nonzero inds */
    pt->nfibs[nmodes-1] = ptnnz;

    pt->fids[nmodes-1] = splatt_malloc(ptnnz * sizeof(**(pt->fids)));
    memcpy(pt->fids[nmodes-1], tt->ind[ct->dim_perm[nmodes-1]] + startnnz,
        ptnnz * sizeof(**(pt->fids)));

    pt->vals = splatt_malloc(ptnnz * sizeof(*(pt->vals)));
    memcpy(pt->vals, tt->vals + startnnz, ptnnz * sizeof(*(pt->vals)));

    /* create fptr entries for the rest of the modes */
    for(idx_t m=0; m < tt->nmodes-1; ++m) {
      p_mk_fptr(ct, tt, t, nnz_ptr, m);
    }
  }

  free(nnz_ptr);
}


/**
* @brief Allocate and fill a CSF tensor.
*
* @param ct The CSF tensor to fill.
* @param tt The coordinate tensor to work from.
* @param mode_type The allocation scheme for the CSF tensor.
* @param mode Which mode we are converting for (if applicable).
* @param splatt_opts Used to determine tiling scheme.
*/
static void p_mk_csf(
  splatt_csf * const ct,
  sptensor_t * const tt,
  csf_mode_type mode_type,
  idx_t const mode,
  double const * const splatt_opts)
{
  ct->nnz = tt->nnz;
  ct->nmodes = tt->nmodes;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ct->dims[m] = tt->dims[m];
  }

  /* get the indices in order */
  csf_find_mode_order(tt->dims, tt->nmodes, mode_type, mode, ct->dim_perm);

  ct->which_tile = splatt_opts[SPLATT_OPTION_TILE];
  switch(ct->which_tile) {
  case SPLATT_NOTILE:
    p_csf_alloc_untiled(ct, tt);
    break;
  case SPLATT_DENSETILE:
    p_csf_alloc_densetile(ct, tt, splatt_opts);
    break;
  default:
    fprintf(stderr, "SPLATT: tiling '%d' unsupported for CSF tensors.\n",
        ct->which_tile);
    break;
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


void csf_free(
  splatt_csf * const csf,
  double const * const opts)
{
  idx_t ntensors = 0;
  splatt_csf_type which = opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {
  case SPLATT_CSF_ONEMODE:
    ntensors = 1;
    break;
  case SPLATT_CSF_TWOMODE:
    ntensors = 2;
    break;
  case SPLATT_CSF_ALLMODE:
    ntensors = csf[0].nmodes;
    break;
  }

  for(idx_t i=0; i < ntensors; ++i) {
    csf_free_mode(csf + i);
  }

  free(csf);
}


void csf_free_mode(
    splatt_csf * const csf)
{
  /* free each tile of sparsity pattern */
  for(idx_t t=0; t < csf->ntiles; ++t) {
    free(csf->pt[t].vals);
    free(csf->pt[t].fids[csf->nmodes-1]);
    for(idx_t m=0; m < csf->nmodes-1; ++m) {
      free(csf->pt[t].fptr[m]);
      free(csf->pt[t].fids[m]);
    }
  }
  free(csf->pt);
}



void csf_find_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  csf_mode_type which,
  idx_t const mode,
  idx_t * const perm_dims)
{
  switch(which) {
  case CSF_SORTED_SMALLFIRST:
    p_order_dims_small(dims, nmodes, perm_dims);
    break;

  case CSF_SORTED_BIGFIRST:
    p_order_dims_large(dims, nmodes, perm_dims);
    break;

  case CSF_INORDER_MINUSONE:
    p_order_dims_inorder(dims, nmodes, mode, perm_dims);
    break;

  case CSF_SORTED_MINUSONE:
    p_order_dims_minusone(dims, nmodes, mode, perm_dims);
    break;

  /* no-op, perm_dims better be set... */
  case CSF_MODE_CUSTOM:
    break;

  default:
    fprintf(stderr, "SPLATT: csf_mode_type '%d' not recognized.\n", which);
    break;
  }
}


size_t csf_storage(
  splatt_csf const * const tensors,
  double const * const opts)
{
  idx_t ntensors = 0;
  splatt_csf_type which_alloc = opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which_alloc) {
  case SPLATT_CSF_ONEMODE:
    ntensors = 1;
    break;
  case SPLATT_CSF_TWOMODE:
    ntensors = 2;
    break;
  case SPLATT_CSF_ALLMODE:
    ntensors = tensors[0].nmodes;
    break;
  }

  size_t bytes = 0;
  for(idx_t m=0; m < ntensors; ++m) {
    splatt_csf const * const ct = tensors + m;
    bytes += ct->nnz * sizeof(*(ct->pt->vals)); /* vals */
    bytes += ct->nnz * sizeof(**(ct->pt->fids)); /* fids[nmodes] */
    bytes += ct->ntiles * sizeof(*(ct->pt)); /* pt */

    for(idx_t t=0; t < ct->ntiles; ++t) {
      csf_sparsity const * const pt = ct->pt + t;

      for(idx_t m=0; m < ct->nmodes-1; ++m) {
        bytes += (pt->nfibs[m]+1) * sizeof(**(pt->fptr)); /* fptr */
        if(pt->fids[m] != NULL) {
          bytes += pt->nfibs[m] * sizeof(**(pt->fids)); /* fids */
        }
      }
    }
  }

  return bytes;
}


splatt_csf * csf_alloc(
  sptensor_t * const tt,
  double const * const opts)
{
  splatt_csf * ret = NULL;

  double * tmp_opts = NULL;
  idx_t last_mode = 0;

  int tmp = 0;

  switch((splatt_csf_type) opts[SPLATT_OPTION_CSF_ALLOC]) {
  case SPLATT_CSF_ONEMODE:
    ret = splatt_malloc(sizeof(*ret));
    p_mk_csf(ret, tt, CSF_SORTED_SMALLFIRST, 0, opts);
    break;

  case SPLATT_CSF_TWOMODE:
    ret = splatt_malloc(2 * sizeof(*ret));
    /* regular CSF allocation */
    p_mk_csf(ret + 0, tt, CSF_SORTED_SMALLFIRST, 0, opts);

    /* make a copy of opts and don't tile the last mode
     * TODO make this configurable? */
    tmp_opts = splatt_default_opts();
    memcpy(tmp_opts, opts, SPLATT_OPTION_NOPTIONS * sizeof(*opts));
    tmp_opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

    /* allocate with no tiling for the last mode */
    last_mode = ret[0].dim_perm[tt->nmodes-1];
    p_mk_csf(ret + 1, tt, CSF_SORTED_MINUSONE, last_mode, tmp_opts);

    free(tmp_opts);
    break;

  case SPLATT_CSF_ALLMODE:
    ret = splatt_malloc(tt->nmodes * sizeof(*ret));
    for(idx_t m=0; m < tt->nmodes; ++m) {
      p_mk_csf(ret + m, tt, CSF_SORTED_MINUSONE, m, opts);
    }
    break;
  }

  return ret;
}


void csf_alloc_mode(
  sptensor_t * const tt,
  csf_mode_type which_ordering,
  idx_t const mode_special,
  splatt_csf * const csf,
  double const * const opts)
{
  p_mk_csf(csf, tt, which_ordering, mode_special, opts);
}


val_t csf_frobsq(
    splatt_csf const * const tensor)
{
  /* accumulate into double to help with some precision loss */
  double norm = 0;
  #pragma omp parallel reduction(+:norm)
  {
    for(idx_t t=0; t < tensor->ntiles; ++t) {
      val_t const * const vals = tensor->pt[t].vals;
      if(vals == NULL) {
        continue;
      }

      idx_t const nnz = tensor->pt[t].nfibs[tensor->nmodes-1];

      #pragma omp for schedule(static) nowait
      for(idx_t n=0; n < nnz; ++n) {
        norm += vals[n] * vals[n];
      }
    }
  } /* end omp parallel */

  return (val_t) norm;
}

