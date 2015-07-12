
#include "mex.h"

#include <stdint.h>
#include <string.h>
#include <splatt.h>

#include "splatt_shared.h"


/******************************************************************************
 * ENTRY FUNCTION
 *****************************************************************************/
void mexFunction(
    int nlhs,
    mxArray * plhs[],
    int nrhs,
    mxArray const * prhs[])
{
  if(nrhs < 3) {
    mexErrMsgTxt("Missing arguments. See 'help splatt_mttkrp' for usage.\n");
    return;
  }

  double * cpd_opts = splatt_default_opts();
  if(nrhs > 1 && mxIsStruct(prhs[nrhs-1])) {
    __parse_opts(prhs[nrhs-1], cpd_opts);
  }


  mxArray * out = mxCreateDoubleMatrix(1, 1, mxREaL);
  if(nlhs > 0) {
    plhs[0] = out;
  }

  splatt_free_opts(cpd_opts);
}
