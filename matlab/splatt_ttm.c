
#include "mex.h"

#include <stdint.h>
#include <string.h>
#include <splatt.h>

#include "splatt_shared.h"

/*
 * [M] = splatt_ttm(X, mats, mode);
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
    mexErrMsgTxt("Missing arguments. See 'help splatt_ttm' for usage.\n");
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

  splatt_idx_t outcols = 1;
  splatt_idx_t nfactors[MAX_NMODES];
  for(m=0; m < nmodes; ++m) {
    mwSize const * matdims = mxGetDimensions(mxGetCell(matcells, m));
    nfactors[m] = (splatt_idx_t) matdims[1];
    mexPrintf("x%lu\n", nfactors[m]);
    if(m != mode) {
      outcols *= nfactors[m];
    }
  }


  /* allocate and transpose matrices */
  splatt_val_t * mats[MAX_NMODES];
  for(m=0; m < nmodes; ++m) {
    splatt_idx_t const dim = tt[0].dims[m];
    mats[m] = (splatt_val_t *) malloc(dim * nfactors[m]
        * sizeof(splatt_val_t));

    /* only bother transposing if we aren't overwriting */
    if(m == mode) {
      continue;
    }

    mxArray const * const curr = mxGetCell(matcells, m);
    double const * const matdata = (double *) mxGetPr(curr);

    for(i=0; i < dim; ++i) {
      for(j=0; j < nfactors[m]; ++j) {
        mats[m][j+(i*nfactors[m])] = (splatt_val_t) matdata[i + (j*dim)];
      }
    }
  }

  splatt_val_t * tenout = (splatt_val_t *) malloc(tt[0].dims[mode] * outcols
      * sizeof(splatt_val_t));

  mexPrintf("allocated (%lu x %lu)\n", tt[0].dims[mode], outcols);

  /* TTM */
  int ret = splatt_ttm(mode, nfactors, tt+mode, mats, tenout, cpd_opts);
  if(ret != SPLATT_SUCCESS) {
    mexPrintf("splatt_ttm returned %d\n", ret);
    goto CLEANUP;
  }

  mexPrintf("computed\n");

  /* allocate and transpose output */
  splatt_idx_t const dim = tt[0].dims[mode];
  mxArray * out = mxCreateDoubleMatrix(dim, outcols, mxREAL);
  double * const outpr = (double *) mxGetPr(out);
  memcpy(outpr, tenout, dim * outcols * sizeof(double));
#if 0
  for(j=0; j < outcols; ++j) {
    for(i=0; i < dim; ++i) {
      outpr[i+(j * dim)] = (double) tenout[j + (i*nfactors[mode])];
    }
  }
#endif

  if(nlhs > 0) {
    plhs[0] = out;
  }

  /* cleanup */
  CLEANUP:
  free(tenout);
  __free_tensor(nrhs, prhs, nmodes, tt);
  splatt_free_opts(cpd_opts);
  for(m=0; m < nmodes; ++m) {
    free(mats[m]);
  }
}


