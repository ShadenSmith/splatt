
#include "mex.h"

#include <stdint.h>
#include <string.h>
#include <splatt.h>

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static uint64_t * __get_uint64_data(
    mxArray const * const mxstruct,
    char const * const field)
{
  return (uint64_t *) mxGetData(mxGetField(mxstruct, 0, field));
}

static double * __get_double_data(
    mxArray const * const mxstruct,
    char const * const field)
{
  return (double *) mxGetData(mxGetField(mxstruct, 0, field));
}


static splatt_csf_t ** __unpack_csf_cell(
    mxArray const * const cell,
    splatt_idx_t * outnmodes)
{
  splatt_idx_t m;
  splatt_csf_t ** tt = NULL;
  splatt_idx_t nmodes = (splatt_idx_t) mxGetNumberOfElements(cell);

  tt = (splatt_csf_t **) malloc(nmodes * sizeof(splatt_csf_t *));

  for(m=0; m < nmodes; ++m) {
    tt[m] = (splatt_csf_t *) malloc(sizeof(splatt_csf_t));

    mxArray const * const curr = mxGetCell(cell, m);

    tt[m]->nmodes = nmodes;
    tt[m]->nnz = *(__get_uint64_data(curr, "nnz"));
    memcpy(tt[m]->dims, __get_uint64_data(curr, "dims"),
        nmodes * sizeof(uint64_t));
    memcpy(tt[m]->dim_perm, __get_uint64_data(curr, "dim_perm"),
        nmodes * sizeof(uint64_t));
    tt[m]->nslcs = *(__get_uint64_data(curr, "nslcs"));
    tt[m]->nfibs = *(__get_uint64_data(curr, "nfibs"));
    tt[m]->sptr = __get_uint64_data(curr, "sptr");
    tt[m]->fptr = __get_uint64_data(curr, "fptr");
    tt[m]->fids = __get_uint64_data(curr, "fids");
    tt[m]->inds = __get_uint64_data(curr, "inds");
    tt[m]->vals = __get_double_data(curr, "vals");

    if(*__get_uint64_data(curr, "has_indmap") == 1) {
      tt[m]->indmap = __get_uint64_data(curr, "indmap");
    }

    tt[m]->tiled = (int) *(__get_uint64_data(curr, "tiled"));
    if(tt[m]->tiled != SPLATT_NOTILE) {
      tt[m]->nslabs = *(__get_uint64_data(curr, "nslabs"));
      tt[m]->slabptr = __get_uint64_data(curr, "slabptr");
      tt[m]->sids = __get_uint64_data(curr, "sids");
    }
  }

  *outnmodes = nmodes;
  return tt;
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
  splatt_idx_t m;
  if(nrhs != 2) {
    mexPrintf("ARG2 must be nfactors\n");
    return;
  }

  if(sizeof(splatt_val_t) != sizeof(double)) {
    mexErrMsgTxt("SPLATT must be compiled with double-precision floats.\n");
  }

  if(sizeof(splatt_idx_t) != sizeof(uint64_t)) {
    mexErrMsgTxt("SPLATT must be compiled with double-precision floats.\n");
  }

  double * cpd_opts = splatt_default_opts();

  /* parse the tensor! */
  splatt_idx_t nmodes;
  splatt_csf_t ** tt;
  if(mxIsCell(prhs[0])) {
    tt = __unpack_csf_cell(prhs[0], &nmodes);
  } else {
    char * fname = (char *) mxArrayToString(prhs[0]);
    tt = splatt_csf_load(fname, &nmodes, cpd_opts);
    mxFree(fname);
  }

  splatt_idx_t const nfactors = (splatt_idx_t) mxGetScalar(prhs[1]);

  /* do the factorization! */
  splatt_kruskal_t factored;
  splatt_cpd(nfactors, nmodes, tt, cpd_opts, &factored);

  /* save dims and free memory */
  splatt_idx_t ttdims[MAX_NMODES];
  for(m=0; m < nmodes; ++m) {
    ttdims[m] = tt[0]->dims[m];
  }

  if(mxIsCell(prhs[0])) {
    /* just clean up pointers, not actual input data */
    for(m=0; m < nmodes; ++m) {
      free(tt[m]);
    }
    free(tt);
  } else {
    /* free parsed/converted data */
    splatt_csf_free(factored.nmodes, tt);
  }

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

