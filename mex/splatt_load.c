
#include "mex.h"

#include "../src/sptensor.h"
#include "../src/ftensor.h"
#include "../src/stats.h"

void mexFunction(
    int nlhs,
    mxArray * plhs[],
    int nrhs,
    mxArray const * prhs[])
{
  int m;
  if(nrhs != 1 || !mxIsChar(prhs[0])) {
    mexPrintf("ARG1 must be a file\n");
    return;
  }

  /* check for file to read */
  char * fname = (char *) mxArrayToString(prhs[0]);

  /* parse the tensor! */
  sptensor_t * tt = tt_read(fname);
  mxFree(fname);

  ftensor_t * ft[MAX_NMODES];
  for(m=0; m < tt->nmodes; ++m) {
    ft[m] = ften_alloc(tt, m, SPLATT_NOTILE);
  }

  char const * keys[] = {"nnz", "dims"};
  mxArray * ret = mxCreateStructMatrix(1, 1, 2, keys);
  mxSetFieldByNumber(ret, 0, 0, mxCreateDoubleScalar(tt->nnz));
  mxSetFieldByNumber(ret, 0, 1, mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, 0));


  /* set output */
  if(nlhs > 0) {
    plhs[0] = ret;
  }

  tt_free(tt);
}

