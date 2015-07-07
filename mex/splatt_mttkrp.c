
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


  mxArray * out = mxCreateDoubleMatrix(1, 1, mxREaL);
  if(nlhs > 0) {
    plhs[0] = out;
  }
}
