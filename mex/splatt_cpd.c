
#include "mex.h"

#include <string.h>
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

  splatt_idx_t const nfactors = (splatt_idx_t) mxGetScalar(prhs[1]);

  /* parse the tensor! */
  char * fname = (char *) mxArrayToString(prhs[0]);
  double * cpd_opts = splatt_default_opts();
  splatt_idx_t nmodes;
  splatt_csf_t ** tt = splatt_csf_load(fname, &nmodes, cpd_opts);
  mxFree(fname);

  /* do the factorization! */
  splatt_kruskal_t factored;
  splatt_cpd(nfactors, nmodes, tt, cpd_opts, &factored);

  /* save dims and free memory */
  splatt_idx_t ttdims[MAX_NMODES];
  for(m=0; m < nmodes; ++m) {
    ttdims[m] = tt[0]->dims[m];
  }
  splatt_csf_free(factored.nmodes, tt);

  mwSize dim = (mwSize) nmodes;
  mxArray * mxLambda = mxCreateDoubleMatrix(nfactors, 1, mxREAL);
  memcpy(mxGetPr(mxLambda), factored.lambda, nfactors * sizeof(double));

  mxArray * matcell = mxCreateCellArray(1, &dim);
  mxSetCell(matcell, 0, mxCreateDoubleMatrix(1, nfactors, mxREAL));
  mxSetCell(matcell, 1, mxCreateDoubleMatrix(2, nfactors, mxREAL));
  mxSetCell(matcell, 2, mxCreateDoubleMatrix(3, nfactors, mxREAL));
  for(m=0; m < nmodes; ++m) {
    splatt_idx_t const nrows = ttdims[m];
    mxSetCell(matcell, m, mxCreateDoubleMatrix(nrows, nfactors, mxREAL));
    memcpy(mxGetPr(mxGetCell(matcell, m)), factored.factors[m],
        nrows * nfactors * sizeof(double));
  }

  char const * keys[] = {"lambda", "U", "fit"};
  mxArray * ret = mxCreateStructMatrix(1, 1, 3, keys);
  mxSetField(ret, 0, "lambda", mxLambda);
  mxSetField(ret, 0, "U", matcell);
  mxSetField(ret, 0, "fit", mxCreateDoubleScalar(factored.fit));

  /* copy output to matlab structure if requested */
  if(nlhs > 0) {
    plhs[0] = ret;
  }

  /* cleanup */
  splatt_free_kruskal(&factored);
  splatt_free_opts(cpd_opts);
}

