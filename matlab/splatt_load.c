
#include "mex.h"

#include <stdint.h>
#include <string.h>
#include <splatt.h>

#include "splatt_shared.h"

/**
* @brief Keys for structure entries.
*/
static char const * csf_keys[] = {
  "nnz",
  "nmodes",
  "dims",
  "dim_perm",
  "which_tile",
  "ntiles",
  "tile_dims",
  "pt"
};


static char const * sparsity_keys[] = {
  "nfibs",
  "fptr",
  "has_fids",
  "fids",
  "vals"
};

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/
static mxArray * p_pack_csf(
    splatt_csf const * const tt,
    double const * const splatt_opts)
{
  splatt_idx_t const nmodes = tt->nmodes;
  splatt_idx_t ntensors;
  splatt_csf_type which = splatt_opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {
  case SPLATT_CSF_ONEMODE:
    ntensors = 1;
    break;
  case SPLATT_CSF_TWOMODE:
    ntensors = 2;
    break;
  case SPLATT_CSF_ALLMODE:
    ntensors = nmodes;
    break;
  }

  /* create splatt_csf matlab struct */
  mxArray * csf = mxCreateCellMatrix(1, (mwSize) ntensors);

  splatt_idx_t t;
  for(t=0; t < ntensors; ++t) {
    uint64_t * data;
    mxArray * curr = mxCreateStructMatrix(1, 1,
        sizeof(csf_keys)/sizeof(csf_keys[0]), csf_keys);

    /* fill in each cell */
    p_mk_uint64(curr, "nnz", 1, &(tt[t].nnz));
    p_mk_uint64(curr, "nmodes", 1, &(nmodes));
    p_mk_uint64(curr, "dims", nmodes, tt[t].dims);
    p_mk_uint64(curr, "dim_perm", nmodes, tt[t].dim_perm);


    /* tiled fields */
    int32_t which = tt[t].which_tile;
    p_mk_int32(curr, "which_tile", 1, &(which));
    p_mk_uint64(curr, "ntiles", 1, &(tt[t].ntiles));
    p_mk_uint64(curr, "tile_dims", 1, tt[t].tile_dims);

    /* sparsity pattern for each tile */
    mxArray * sparsities = mxCreateCellMatrix(1, (mwSize) tt[t].ntiles);

    splatt_idx_t tile;
    for(tile=0; tile < tt[t].ntiles; ++tile) {
      mxArray * curr_tile = mxCreateStructMatrix(1, 1,
          sizeof(sparsity_keys)/sizeof(sparsity_keys[0]), sparsity_keys);

      csf_sparsity const * const pt = &(tt[t].pt[tile]);

      p_mk_uint64(curr_tile, "nfibs", nmodes, pt->nfibs);
      if(pt->nfibs[0] == 0) {
        mxSetCell(sparsities, tile, curr_tile);
        continue;
      }

      if(pt->nfibs[nmodes-1] > 0) {
        p_mk_double(curr_tile, "vals", pt->nfibs[nmodes-1], pt->vals);
      }

      /* copy fptrs */
      mxArray * mxfptrs = mxCreateCellMatrix(1, (mwSize) nmodes-1);
      splatt_idx_t m;
      for(m=0; m < nmodes-1; ++m) {
        mxArray * fp = mxCreateNumericMatrix(1, pt->nfibs[m]+1, mxUINT64_CLASS, mxREAL);
        memcpy(mxGetData(fp), pt->fptr[m], (1 + pt->nfibs[m]) * sizeof(uint64_t));
        mxSetCell(mxfptrs, m, fp);
      }

      mxSetField(curr_tile, 0, "fptr", mxfptrs);

      /* copy fids */
      mxArray * mxfids = mxCreateCellMatrix(1, (mwSize) nmodes);

      mxArray * has_fids = mxCreateNumericMatrix(1, nmodes, mxINT32_CLASS,
          mxREAL);
      int32_t has[SPLATT_MAX_NMODES];

      for(m=0; m < nmodes; ++m) {
        has[m] = (pt->fids[m] != NULL);

        if(pt->fids[m] != NULL) {
          mxArray * fi = mxCreateNumericMatrix(1, pt->nfibs[m], mxUINT64_CLASS, mxREAL);
          memcpy(mxGetData(fi), pt->fids[m], pt->nfibs[m] * sizeof(uint64_t));
          mxSetCell(mxfids, m, fi);
        }
      }
      mxSetField(curr_tile, 0, "fids", mxfids);

      memcpy(mxGetData(has_fids), has, nmodes * sizeof(int32_t));
      mxSetField(curr_tile, 0, "has_fids", has_fids);

      /* store new sparsity pattern */
      mxSetCell(sparsities, tile, curr_tile);
    }

    mxSetField(curr, 0, "pt", sparsities);

    /* store struct */
    mxSetCell(csf, t, curr);
  }

  return csf;
}

/******************************************************************************
 * ENTRY FUNCTION
 *****************************************************************************/
void mexFunction(
    int nlhs,
    mxArray * plhs[],
    int nrhs,
    mxArray const * prhs[])
{
  if(nrhs == 0) {
    mexErrMsgTxt("ARG1 must be a file or sptensor\n");
  }

  double * cpd_opts = splatt_default_opts();

  splatt_idx_t nmodes;
  splatt_csf * tt = p_parse_tensor(nrhs, prhs, &nmodes, cpd_opts);
  if(tt == NULL) {
    splatt_free_opts(cpd_opts);
    return;
  }

  mxArray * csf = p_pack_csf(tt, cpd_opts);

  p_free_tensor(nrhs, prhs, tt, cpd_opts);

  splatt_free_opts(cpd_opts);
  if(nlhs > 0) {
    plhs[0] = csf;
  }
}

