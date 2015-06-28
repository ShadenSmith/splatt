
#include "mex.h"

#include <splatt.h>
#include "../src/sptensor.h"
#include "../src/ftensor.h"

void mexFunction(
    int nlhs,
    mxArray * plhs[],
    int nrhs,
    mxArray const * prhs[])
{
  if(nrhs != 2) {
    mexPrintf("ARG2 must be nfactors\n");
    return;
  }

  if(sizeof(val_t) != sizeof(double)) {
    mexErrMsgTxt("SPLATT must be compiled with double-precision floats.\n");
  }

  /* parse the tensor! */
  char * fname = (char *) mxArrayToString(prhs[0]);
  sptensor_t * tt = tt_read(fname);

  idx_t const nfactors = (idx_t) mxGetScalar(prhs[1]);
  mxArray * mxLambda = mxCreateDoubleMatrix(nfactors, 1, mxREAL);
  val_t * lambda = mxGetPr(mxLambda);

  double * cpd_opts = splatt_default_opts();
  cpd_opts[SPLATT_OPTION_NTHREADS] = 2;

  mxArray * mxMats[MAX_NMODES];
  double * mats[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mxMats[m] = mxCreateDoubleMatrix(tt->dims[m], nfactors, mxREAL);
    mats[m] = (val_t *) mxGetPr(mxMats[m]);
  }

  plhs[0] = mxLambda;
  /* do the factorization! */
  splatt_cpd(nfactors, tt->nmodes, tt->nnz, tt->ind, tt->vals, mats, lambda,
      cpd_opts);

  /* clean up */
  tt_free(tt);
  mxFree(fname);
  splatt_free_opts(cpd_opts);
}

