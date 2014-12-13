

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "reorder.h"

#include "sptensor.h"
#include "ftensor.h"
#include "io.h"
#include "sort.h"

#include <assert.h>


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/
static void __reorder_slices(
  sptensor_t * const tt,
  ftensor_t const * const ft,
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const * const uncut,
  idx_t const mode)
{
  /* build map of fiber -> slice */
  idx_t const nslices = ft->dims[mode];
  idx_t const nfibs = ft->nfibs[mode];
  idx_t * slice = (idx_t *) malloc(nfibs * sizeof(idx_t));

  idx_t * sliceperm = (idx_t *) malloc(nslices * sizeof(idx_t));
  idx_t const * const sptr = ft->sptr[mode];
  for(idx_t s=0; s < nslices; ++s) {
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      slice[f] = s;
    }
    /* mark perm as incomplete */
    sliceperm[s] = nslices;
  }

  idx_t * pptr = NULL;
  idx_t * plookup = NULL;
  build_pptr(parts, nparts, nfibs, &pptr, &plookup);

  idx_t sliceptr = 0;
  idx_t uncutptr = 0;

  /* order all uncut slices first */
  for(idx_t p=0; p < nparts; ++p) {
    uncutptr = 0;
    /* for each fiber in partition */
    for(idx_t j=pptr[p]; j < pptr[p+1]; ++j) {
      idx_t const fib = plookup[j];
      idx_t const s = slice[fib];
      /* move to uncut slice (or past it) */
      while(uncut[uncutptr] < s) {
        ++uncutptr;
      }

      /* mark s if it is uncut and not already marked */
      if(uncut[uncutptr] == s && sliceperm[s] == nslices) {
        sliceperm[s] = sliceptr++;
      }
    }
  }

  /* place untouched slices at end of permutation */
  for(idx_t s=0; s < nslices; ++s) {
    if(sliceperm[s] == nslices) {
      sliceperm[s] = sliceptr++;
    }
  }
  assert(sliceptr == nslices);

  /* now do actual reordering */
  idx_t const nnz = tt->nnz;
  idx_t * const ind = tt->ind[mode];
  for(idx_t n=0; n < nnz; ++n) {
    ind[n] = sliceperm[ind[n]];
  }

  free(pptr);
  free(plookup);
  free(sliceperm);
  free(slice);
}

static void __reorder_fibs(
  sptensor_t * const tt,
  ftensor_t const * const ft,
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const * const uncut,
  idx_t const mode)
{
  /* build map of fiber -> fid */
  idx_t const nfids = ft->dims[mode];
  idx_t const nfibs = ft->nfibs[mode];
  idx_t * fid = (idx_t *) malloc(nfibs * sizeof(idx_t));

  idx_t * fidperm = (idx_t *) malloc(nfids * sizeof(idx_t));
  idx_t const * const sptr = ft->sptr[mode];
  for(idx_t s=0; s < nfids; ++s) {
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      fid[f] = ft->fids[mode][f];
    }
    /* mark perm as incomplete */
    fidperm[s] = nfids;
  }

  idx_t * pptr = NULL;
  idx_t * plookup = NULL;
  build_pptr(parts, nparts, nfibs, &pptr, &plookup);

  idx_t fidptr = 0;
  idx_t uncutptr = 0;

  /* order all uncut fids first */
  for(idx_t p=0; p < nparts; ++p) {
    uncutptr = 0;
    /* for each fiber in partition */
    for(idx_t j=pptr[p]; j < pptr[p+1]; ++j) {
      idx_t const fib = plookup[j];
      idx_t const s = fid[fib];
      /* move to uncut slice (or past it) */
      while(uncut[uncutptr] < s) {
        ++uncutptr;
      }

      /* mark s if it is uncut and not already marked */
      if(uncut[uncutptr] == s && fidperm[s] == nfids) {
        fidperm[s] = fidptr++;
      }
    }
  }

  /* place untouched slices at end of permutation */
  for(idx_t s=0; s < nfids; ++s) {
    if(fidperm[s] == nfids) {
      fidperm[s] = fidptr++;
    }
  }
  assert(fidptr == nfids);

  /* now do actual reordering */
  idx_t const nnz = tt->nnz;
  idx_t * const ind = tt->ind[ft->dim_perms[mode][1]];
  for(idx_t n=0; n < nnz; ++n) {
    ind[n] = fidperm[ind[n]];
  }

  free(pptr);
  free(plookup);
  free(fidperm);
  free(fid);
}


static void __perm_hgraph(
  sptensor_t * const tt,
  idx_t const mode,
  char const * const pfile)
{
  ftensor_t * ft = ften_alloc(tt);
  idx_t const nvtxs = ft->nfibs[mode];
  idx_t nhedges = 0;
  for(idx_t m=0; m < ft->nmodes; ++m) {
    nhedges += ft->dims[m];
  }

  idx_t nparts = 0;
  idx_t * parts = part_read(pfile, nvtxs, &nparts);
  hgraph_t * hg = hgraph_fib_alloc(ft, mode);

  printf("nvtxs: "SS_IDX" nhedges: "SS_IDX"  nparts: "SS_IDX"\n",
    nvtxs, nhedges, nparts);

  idx_t ncut = 0;
  idx_t * uncuts  = hgraph_uncut(hg, parts, &ncut);
  printf("cut: "SS_IDX"  notcut: "SS_IDX"\n", nhedges - ncut, ncut);

  /* track number of uncut slices, fids, and inds */
  idx_t nslices = 0;
  idx_t nfibs = 0;
  idx_t ninds = 0;
  for(idx_t h=0; h < ncut; ++h) {
    if(uncuts[h] < ft->dims[mode]) {
      ++nslices;
    } else if(uncuts[h] < (ft->dims[mode] + ft->dims[ft->dim_perms[mode][1]])) {
      ++nfibs;
    } else {
      ++ninds;
    }
  }

  printf("nslices: "SS_IDX"  nfibs: "SS_IDX"  ninds: "SS_IDX"\n",
    nslices, nfibs, ninds);

  /* reorder slices */
  __reorder_slices(tt, ft, parts, nparts, uncuts, mode);
  __reorder_fibs(tt, ft, parts, nparts, uncuts, mode);

  tt_sort(tt, mode, NULL);
  tt_write(tt, "permed.tns");

  free(uncuts);
  free(parts);
  hgraph_free(hg);
  ften_free(ft);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_perm(
  sptensor_t * const tt,
  splatt_perm_type const type,
  idx_t const mode,
  char const * const pfile)
{
  switch(type) {
  case PERM_GRAPH:
    break;
  case PERM_HGRAPH:
    if(pfile == NULL) {
      fprintf(stderr, "SPLATT: permutation file must be supplied for now.\n");
      return;
    }
    __perm_hgraph(tt, mode, pfile);
    break;
  default:
    break;
  }
}


void build_pptr(
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const nvtxs,
  idx_t ** ret_pptr,
  idx_t ** ret_plookup)
{
  /* pptr marks the size of each partition (in vtxs, not nnz) */
  idx_t * pptr = (idx_t *) calloc(nparts+1, sizeof(idx_t));
  for(idx_t v=0; v < nvtxs; ++v) {
    pptr[1+parts[v]]++;
  }

  /* prefix sum of pptr */
  idx_t saved = pptr[1];
  pptr[1] = 0;
  for(idx_t p=2; p <= nparts; ++p) {
    idx_t tmp = pptr[p];
    pptr[p] = pptr[p-1] + saved;
    saved = tmp;
  }

  idx_t * plookup = (idx_t *) malloc(nvtxs * sizeof(idx_t));
  for(idx_t f=0; f < nvtxs; ++f) {
    idx_t const index = pptr[1+parts[f]]++;
    plookup[index] = f;
  }

  *ret_pptr = pptr;
  *ret_plookup = plookup;
}



/******************************************************************************
 * MATRIX REORDER FUNCTIONS
 *****************************************************************************/
matrix_t * perm_matrix(
  matrix_t const * const mat,
  idx_t const * const perm,
  matrix_t * retmat)
{
  idx_t const I = mat->I;
  idx_t const J = mat->J;

  if(retmat == NULL) {
    retmat = (matrix_t *) malloc(sizeof(matrix_t));
    retmat->I = I;
    retmat->J = J;
    retmat->vals = (val_t *) malloc(I * J * sizeof(val_t));
  }

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      retmat->vals[j + (i*J)] = mat->vals[j + (perm[i] * J)];
    }
  }

  return retmat;
}



