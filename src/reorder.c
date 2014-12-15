

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
  idx_t const nuncut,
  permutation_t * const perm,
  idx_t const mode)
{
  /* build map of fiber -> slice */
  idx_t const nslices = ft->dims[mode];
  idx_t const nfibs = ft->nfibs[mode];
  idx_t * slice = (idx_t *) malloc(nfibs * sizeof(idx_t));

  idx_t * const sliceperm  = perm->perms[mode];
  idx_t * const sliceiperm = perm->iperms[mode];

  idx_t const * const sptr = ft->sptr[mode];
  for(idx_t s=0; s < nslices; ++s) {
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      slice[f] = s;
    }
    /* mark perm as incomplete */
    sliceperm[s] = nslices;
    sliceiperm[s] = nslices;
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
      while(uncutptr < nuncut && uncut[uncutptr] < s) {
        ++uncutptr;
      }
      if(uncutptr == nuncut) {
        break;
      }

      /* mark s if it is uncut and not already marked */
      if(uncut[uncutptr] == s && sliceperm[s] == nslices) {
        sliceiperm[sliceptr] = s;
        sliceperm[s] = sliceptr++;
      }
    }
  }

  printf("placed: %lu\n", sliceptr);
  /* place untouched slices at end of permutation */
  for(idx_t s=0; s < nslices; ++s) {
    if(sliceperm[s] == nslices) {
      sliceiperm[sliceptr] = s;
      sliceperm[s] = sliceptr++;
    }
  }
  assert(sliceptr == nslices);

  free(pptr);
  free(plookup);
  free(slice);
}

static void __reorder_fibs(
  sptensor_t * const tt,
  ftensor_t const * const ft,
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const * const uncut,
  idx_t const nuncut,
  permutation_t * const perm,
  idx_t const mode)
{
  idx_t const pm = ft->dim_perms[mode][1];
  idx_t const nslices = ft->dims[mode];
  idx_t const nfids = ft->dims[pm];
  idx_t const nfibs = ft->nfibs[mode];
  idx_t * fid = (idx_t *) malloc(nfibs * sizeof(idx_t));

  idx_t * const fidperm  = perm->perms[pm];
  idx_t * const fidiperm = perm->iperms[pm];
  idx_t const * const sptr = ft->sptr[mode];

  /* build map of fiber -> fid */
  for(idx_t f=0; f < nfibs; ++f) {
    fid[f] = ft->fids[mode][f];
  }

  for(idx_t f=0; f < nfids; ++f) {
    /* mark perm as incomplete */
    fidperm[f] = nfids;
    fidiperm[f] = nfids;
  }

  idx_t * pptr = NULL;
  idx_t * plookup = NULL;
  build_pptr(parts, nparts, nfibs, &pptr, &plookup);

  idx_t fidptr = 0;
  idx_t uncutptr = 0;
  idx_t uncutstart = 0;
  while(uncut[uncutstart] < nslices) {
    ++uncutstart;
  }

  /* order all uncut fids first */
  for(idx_t p=0; p < nparts; ++p) {
    /* for each fiber in partition */
    for(idx_t j=pptr[p]; j < pptr[p+1]; ++j) {
      uncutptr = uncutstart;
      idx_t const fib = plookup[j];
      idx_t const s = fid[fib];
      /* move to uncut slice (or past it) */
      while(uncutptr < nuncut && uncut[uncutptr] < (s + nslices)) {
        ++uncutptr;
      }
      if(uncutptr == nuncut) {
        break;
      }

      /* mark s if it is uncut and not already marked */
      if(uncut[uncutptr] == (s + nslices) && fidperm[s] == nfids) {
        fidiperm[fidptr] = s;
        fidperm[s] = fidptr++;
      }
    }
  }

  /* place untouched slices at end of permutation */
  printf("placed: %lu\n", fidptr);
  for(idx_t s=0; s < nfids; ++s) {
    if(fidperm[s] == nfids) {
      fidiperm[fidptr] = s;
      fidperm[s] = fidptr++;
    }
  }
  assert(fidptr == nfids);

  free(pptr);
  free(plookup);
  free(fid);
}

static void __reorder_inds(
  sptensor_t * const tt,
  ftensor_t const * const ft,
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const * const uncut,
  idx_t const nuncut,
  permutation_t * const perm,
  idx_t const mode)
{
  idx_t const pm = ft->dim_perms[mode][2];
  idx_t * const indperm  = perm->perms[pm];
  idx_t * const indiperm = perm->iperms[pm];

  idx_t const nslices = ft->dims[mode];
  idx_t const nfids = ft->dims[ft->dim_perms[mode][1]];
  idx_t const ninds = ft->dims[pm];
  idx_t const nfibs = ft->nfibs[mode];
  idx_t const * const fptr = ft->fptr[mode];
  idx_t const * const inds = ft->inds[mode];


  /* mark perm as incomplete */
  for(idx_t f=0; f < ninds; ++f) {
    indperm[f] = ninds;
    indiperm[f] = ninds;
  }

  idx_t * pptr = NULL;
  idx_t * plookup = NULL;
  build_pptr(parts, nparts, nfibs, &pptr, &plookup);

  idx_t fidptr = 0;
  idx_t uncutptr = 0;
  idx_t uncutstart = 0;
  while(uncut[uncutstart] < nslices + nfids) {
    ++uncutstart;
  }

  idx_t indptr = 0;

  /* order all uncut fids first */
  for(idx_t p=0; p < nparts; ++p) {
    /* for each fiber in partition */
    for(idx_t j=pptr[p]; j < pptr[p+1]; ++j) {
      uncutptr = uncutstart;

      /* traverse fiber and mark inds */
      idx_t const fib = plookup[j];
      for(idx_t j=fptr[fib]; j < fptr[fib+1]; ++j) {
        idx_t const s = inds[j];

        while(uncutptr < nuncut && uncut[uncutptr] < (s + nslices + nfids)) {
          ++uncutptr;
        }
        if(uncutptr == nuncut) {
          break;
        }
        /* mark s if it is uncut and not already marked */
        if(uncut[uncutptr] == (s + nslices + nfids) && indperm[s] == ninds) {
          indiperm[indptr] = s;
          indperm[s] = indptr++;
        }
      }
    }
  }

  /* place untouched slices at end of permutation */
  printf("placed: %lu\n", indptr);
  for(idx_t s=0; s < ninds; ++s) {
    if(indperm[s] == ninds) {
      indiperm[indptr] = s;
      indperm[s] = indptr++;
    }
  }
  assert(indptr == ninds);

  free(pptr);
  free(plookup);
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
permutation_t * tt_perm(
  sptensor_t * const tt,
  splatt_perm_type const type,
  idx_t const mode,
  char const * const pfile)
{
  permutation_t * perm = NULL;
  switch(type) {
  case PERM_GRAPH:
    break;
  case PERM_HGRAPH:
    if(pfile == NULL) {
      fprintf(stderr, "SPLATT: permutation file must be supplied for now.\n");
      exit(1);
    }
    ftensor_t * ft = ften_alloc(tt);
    idx_t nparts = 0;
    idx_t * parts = part_read(pfile, ft->nfibs[mode], &nparts);
    perm = perm_hgraph(tt, ft, parts, nparts, mode);
    free(parts);
    ften_free(ft);
    break;
  default:
    break;
  }

  return perm;
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


void perm_apply(
  sptensor_t * const tt,
  idx_t ** perm)
{
  idx_t const nnz = tt->nnz;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t * const ind = tt->ind[m];
    idx_t const * const p = perm[m];
    for(idx_t n=0; n < nnz; ++n) {
      ind[n] = p[ind[n]];
    }
  }
}

permutation_t * perm_hgraph(
  sptensor_t * const tt,
  ftensor_t const * const ft,
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const mode)
{
  permutation_t * perm = perm_alloc(tt->dims, tt->nmodes);
  hgraph_t * hg = hgraph_fib_alloc(ft, mode);
  idx_t const nvtxs = ft->nfibs[mode];
  idx_t nhedges = 0;
  for(idx_t m=0; m < ft->nmodes; ++m) {
    nhedges += ft->dims[m];
  }

  printf("nvtxs: "SS_IDX" nhedges: "SS_IDX"  nparts: "SS_IDX"\n",
    nvtxs, nhedges, nparts);

  idx_t ncut = 0;
  idx_t * uncuts  = hgraph_uncut(hg, parts, &ncut);
  hgraph_free(hg);
  printf("cut: "SS_IDX"  notcut: "SS_IDX"\n", nhedges - ncut, ncut);

  idx_t nslices = 0;
  idx_t nfibs = 0;
  idx_t ninds = 0;
  for(idx_t n=0; n < ncut; ++n) {
    if(uncuts[n] < ft->dims[mode]) {
      ++nslices;
    } else if(uncuts[n] < ft->dims[mode] + ft->dims[ft->dim_perms[mode][1]]) {
      ++nfibs;
    } else {
      ++ninds;
    }
  }
  printf("slices: "SS_IDX"  fibs: "SS_IDX"  inds: "SS_IDX"\n", nslices, nfibs, ninds);

  for(idx_t m=0; m < tt->nmodes; ++m) {
    for(idx_t n=0; n < tt->dims[m]; ++n) {
      perm->perms[m][n] = n;
      perm->iperms[m][n] = n;
    }
  }

  __reorder_slices(tt, ft, parts, nparts, uncuts, ncut, perm, mode);
  __reorder_fibs(tt, ft, parts, nparts, uncuts, ncut, perm, mode);
  __reorder_inds(tt, ft, parts, nparts, uncuts, ncut, perm, mode);

#if 0
  tt_sort(tt, mode, NULL);
  tt_write(tt, "orig.tns");
#endif

  /* actually apply permutation */
  perm_apply(tt, perm->perms);

#if 0
  tt_sort(tt, mode, NULL);
  tt_write(tt, "permd.tns");
  perm_apply(tt, perm->iperms);
  tt_sort(tt, mode, NULL);
  tt_write(tt, "ipermd.tns");
#endif

  free(uncuts);
  return perm;
}

permutation_t * perm_graph(
  sptensor_t * const tt,
  idx_t const * const parts)
{
  permutation_t * perm = perm_alloc(tt->dims, tt->nmodes);

  return perm;
}

permutation_t * perm_alloc(
  idx_t const * const dims,
  idx_t const nmodes)
{
  permutation_t * perm = (permutation_t *) malloc(sizeof(permutation_t));

  for(idx_t m=0; m < nmodes; ++m) {
    perm->perms[m]  = (idx_t *) malloc(dims[m] * sizeof(idx_t));
    perm->iperms[m] = (idx_t *) malloc(dims[m] * sizeof(idx_t));
  }
  for(idx_t m=nmodes; m < MAX_NMODES; ++m ) {
    perm->perms[m]  = NULL;
    perm->iperms[m] = NULL;
  }

  return perm;
}

void perm_free(
  permutation_t * perm)
{
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    free(perm->perms[m]);
    free(perm->iperms[m]);
  }
  free(perm);
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

  /* allocate retmat if it isn't supplied */
  if(retmat == NULL) {
    retmat = mat_alloc(I, J);
    retmat->rowmajor = mat->rowmajor;
  }

  /* support rowmajor and colmajor */
  if(mat->rowmajor) {
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        retmat->vals[j + (perm[i]*J)] = mat->vals[j + (i * J)];
      }
    }
  } else {
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        retmat->vals[perm[i] + (j*I)] = mat->vals[i + (j * I)];
      }
    }
  }

  return retmat;
}



