
#include "mex.h"

#include <splatt.h>

void mexFunction(
    int nlhs,
    mxArray * plhs[],
    int nrhs,
    mxArray const * prhs[])
{
  int m;
  if(nrhs != 2) {
    mexPrintf("ARG2 must be nfactors\n");
    return;
  }

  if(sizeof(splatt_val_t) != sizeof(double)) {
    mexErrMsgTxt("SPLATT must be compiled with double-precision floats.\n");
  }

  /* parse the tensor! */
  char * fname = (char *) mxArrayToString(prhs[0]);

  splatt_idx_t nmodes;
  splatt_idx_t nnz;
  splatt_idx_t * dims;
  splatt_idx_t ** inds;
  splatt_val_t * vals;

  splatt_load(fname, &nmodes, &dims, &nnz, &inds, &vals);

  splatt_idx_t const nfactors = (splatt_idx_t) mxGetScalar(prhs[1]);
  mxArray * mxLambda = mxCreateDoubleMatrix(nfactors, 1, mxREAL);
  splatt_val_t * lambda = mxGetPr(mxLambda);

  double * cpd_opts = splatt_default_opts();
  cpd_opts[SPLATT_OPTION_NTHREADS] = 2;

  mxArray * mxMats[MAX_NMODES];
  double * mats[MAX_NMODES];
  for(m=0; m < nmodes; ++m) {
    mxMats[m] = mxCreateDoubleMatrix(dims[m], nfactors, mxREAL);
    mats[m] = (splatt_val_t *) mxGetPr(mxMats[m]);
  }

  /* do the factorization! */
  splatt_cpd(nfactors, nmodes, nnz, inds, vals, mats, lambda, cpd_opts);

  if(nlhs > 0) {
    plhs[0] = mxLambda;
  }

  /* clean up */
  for(m=0; m < nmodes; ++m) {
    free(inds[m]);
  }
  free(inds);
  free(vals);
  free(dims);
  mxFree(fname);
  splatt_free_opts(cpd_opts);
}

