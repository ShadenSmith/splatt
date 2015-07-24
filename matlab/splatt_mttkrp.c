
#include "mex.h"

#include <stdint.h>
#include <string.h>
#include <splatt.h>

#include "splatt_shared.h"

/*
 * [M] = splatt_mttkrp(X, mats, mode);
*/

/******************************************************************************
 * ENTRY FUNCTION
 *****************************************************************************/
void mexFunction(
    int nlhs,
    mxArray * plhs[],
    int nrhs,
    mxArray const * prhs[])
{
  splatt_idx_t m;
  splatt_idx_t i,j;

  if(nrhs < 3) {
    mexErrMsgTxt("Missing arguments. See 'help splatt_mttkrp' for usage.\n");
    return;
  }

  double * cpd_opts = splatt_default_opts();
  if(nrhs > 1 && mxIsStruct(prhs[nrhs-1])) {
    __parse_opts(prhs[nrhs-1], cpd_opts);
  }

  mxArray const * matcells = prhs[1];

  splatt_idx_t nmodes;
  splatt_csf_t * tt = __parse_tensor(nrhs, prhs, &nmodes,cpd_opts);

  splatt_idx_t const mode = (splatt_idx_t) mxGetScalar(prhs[2]) - 1;

  mwSize const * matdims = mxGetDimensions(mxGetCell(matcells, nmodes-1));
  splatt_idx_t const nfactors = (splatt_idx_t) matdims[1];

  /* allocate and transpose matrices */
  splatt_val_t * mats[MAX_NMODES];
  for(m=0; m < nmodes; ++m) {
    splatt_idx_t const dim = tt[0].dims[m];
    mats[m] = (splatt_val_t *) malloc(dim * nfactors * sizeof(splatt_val_t));

    /* only bother transposing if we aren't overwriting */
    if(m == mode) {
      continue;
    }

    mxArray const * const curr = mxGetCell(matcells, m);
    double const * const matdata = (double *) mxGetPr(curr);

    for(i=0; i < dim; ++i) {
      for(j=0; j < nfactors; ++j) {
        mats[m][j+(i*nfactors)] = (splatt_val_t) matdata[i + (j*dim)];
      }
    }
  }

  /* MTTKRP */
  int ret = splatt_mttkrp(mode, nfactors, tt+mode, mats, mats[mode], cpd_opts);
  if(ret != SPLATT_SUCCESS) {
    mexPrintf("splatt_mttkrp returned %d\n", ret);
    goto CLEANUP;
  }

  /* allocate and transpose output */
  splatt_idx_t const dim = tt[0].dims[mode];
  mxArray * out = mxCreateDoubleMatrix(dim, nfactors, mxREAL);
  double * const outpr = (double *) mxGetPr(out);
  splatt_val_t const * const matpr = mats[mode];
  for(j=0; j < nfactors; ++j) {
    for(i=0; i < dim; ++i) {
      outpr[i+(j * dim)] = (double) matpr[j + (i*nfactors)];
    }
  }

  if(nlhs > 0) {
    plhs[0] = out;
  }

  /* cleanup */
  CLEANUP:
  __free_tensor(nrhs, prhs, nmodes, tt);
  splatt_free_opts(cpd_opts);
  for(m=0; m < nmodes; ++m) {
    free(mats[m]);
  }
}
