
#include "mex.h"

#include <stdint.h>
#include <string.h>
#include <splatt.h>

#include <stdio.h>

#include "splatt_shared.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

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
  if(nrhs < 2) {
    mexErrMsgTxt("ARG2 must be nfactors\n");
    return;
  }

  if(sizeof(splatt_val_t) != sizeof(double)) {
    mexErrMsgTxt("SPLATT must be compiled with double-precision floats.\n");
    return;
  }

  if(sizeof(splatt_idx_t) != sizeof(uint64_t)) {
    mexErrMsgTxt("SPLATT must be compiled with 64-bit ints.\n");
    return;
  }

  double * cpd_opts = splatt_default_opts();

  if(nrhs > 1 && mxIsStruct(prhs[nrhs-1])) {
    p_parse_opts(prhs[nrhs-1], cpd_opts);
  }

  /* parse the tensor! */
  splatt_idx_t nmodes;
  splatt_csf * tt = p_parse_tensor(nrhs, prhs, &nmodes, cpd_opts);
  if(tt == NULL) {
    splatt_free_opts(cpd_opts);
    return;
  }

  splatt_idx_t const nfactors = (splatt_idx_t) mxGetScalar(prhs[1]);

  /* do the factorization! */
  splatt_kruskal factored;
  int err = splatt_cpd_als(tt, nfactors, cpd_opts, &factored);
  if(err != SPLATT_SUCCESS) {
    mexErrMsgTxt("splatt_cpd_als returned error.\n");
    return;
  }

  p_free_tensor(nrhs, prhs, tt, cpd_opts);

  mxArray * mxLambda = mxCreateDoubleMatrix(nfactors, 1, mxREAL);
  memcpy(mxGetPr(mxLambda), factored.lambda, nfactors * sizeof(double));


  mxArray * matcell = mxCreateCellMatrix(1, nmodes);
  for(m=0; m < nmodes; ++m) {
    splatt_idx_t const nrows = factored.dims[m];

    mxArray * curr_mat = mxCreateDoubleMatrix(nrows, nfactors, mxREAL);

    /* we have to transpose due to column-major ordering in matlab */
    double * const restrict mxpr = mxGetPr(curr_mat);
    double const * const restrict sppr = factored.factors[m];
    splatt_idx_t i, j;
    for(j=0; j < nfactors; ++j) {
      for(i=0; i < nrows; ++i) {
        mxpr[i + (j*nrows)] = sppr[j + (i*nfactors)];
      }
    }

    /* store in matcell */
    mxSetCell(matcell, m, curr_mat);
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

