
#include "mex.h"

#include <stdint.h>
#include <string.h>
#include <splatt.h>

#include "splatt_shared.h"

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static uint64_t * __get_data(
    mxArray const * const mxstruct,
    char const * const field)
{
  return mxGetData(mxGetField(mxstruct, 0, field));
}

static void __mk_int32(
    mxArray * const mxstruct,
    char const * const field,
    splatt_idx_t const len,
    int32_t const * const vals)
{
  mxSetField(mxstruct, 0, field,
      mxCreateNumericMatrix(1, len, mxINT32_CLASS, mxREAL));
  memcpy(__get_data(mxstruct, field), vals, len * sizeof(int32_t));
}

static void __mk_uint64(
    mxArray * const mxstruct,
    char const * const field,
    splatt_idx_t const len,
    splatt_idx_t const * const vals)
{
  mxSetField(mxstruct, 0, field,
      mxCreateNumericMatrix(1, len, mxUINT64_CLASS, mxREAL));
  memcpy(__get_data(mxstruct, field), vals, len * sizeof(uint64_t));
}

static void __mk_double(
    mxArray * const mxstruct,
    char const * const field,
    splatt_idx_t const len,
    splatt_val_t const * const vals)
{
  mxSetField(mxstruct, 0, field,
      mxCreateDoubleMatrix(1, len, mxREAL));
  memcpy(__get_data(mxstruct, field), vals, len * sizeof(double));
}


/**
* @brief Keys for structure entries.
*/
static char const * keys[] = {
  "nnz",
  "nmodes",
  "dims",
  "dim_perm",
  "nslcs",
  "nfibs",
  "sptr",
  "fptr",
  "fids",
  "inds",
  "vals",
  "has_indmap",  /* marks whether indmap is NULL or not after packing */
  "indmap",
  "tiled",
  "nslabs",
  "slabptr",
  "sids"
};


/**
* @brief Convert an inds matrix and vals vector (from sptensor) to CSF.
*
* @param mat_inds An (nnz x nmodes) matrix of indices.
* @param mat_vals An (nnz x 1) vector of values.
* @param nmodes A point to to be set specifying the number of found modes.
* @param cpd_opts SPLATT options array.
*
* @return An array of splatt_csf_t* tensors.
*/
static splatt_csf_t ** __convert_sptensor(
    mxArray const * const mat_inds,
    mxArray const * const mat_vals,
    splatt_idx_t * const nmodes,
    double const * const cpd_opts)
{
  splatt_idx_t m;

  /* parse from tensor toolbox's sptensor */
  mwSize * dims = mxGetDimensions(mat_inds);
  splatt_idx_t nnz = dims[0];
  *nmodes = dims[1];

  /* allocate extra tensor for re-arranging */
  splatt_val_t * vals = (splatt_val_t *) malloc(nnz * sizeof(splatt_val_t));
  splatt_idx_t * inds[MAX_NMODES];
  for(m=0; m < *nmodes; ++m) {
    inds[m] = (splatt_idx_t *) malloc(nnz * sizeof(splatt_idx_t));
  }

  /* subs will be a column-major matrix of size (nnz x nmodes) */
  double const * const mxinds = mxGetPr(mat_inds);
  double const * const mxvals = mxGetPr(mat_vals);

  /* copy indices to inds and adjust for 1-indexing */
  splatt_idx_t n;
  for(n=0; n < nnz; ++n) {
    for(m=0; m < *nmodes; ++m) {
      inds[m][n] = (splatt_idx_t) mxinds[n + (m*nnz)] - 1;
    }
    vals[n] = (splatt_val_t) mxvals[n];
  }

  splatt_csf_t ** tt = splatt_csf_convert(*nmodes, nnz, inds, vals, cpd_opts);

  for(m=0; m < *nmodes; ++m) {
    free(inds[m]);
  }
  free(vals);

  return tt;
}

static mxArray * __pack_csf(
    splatt_csf_t ** tt,
    splatt_idx_t const nmodes)
{
  /* create splatt_csf_t matlab struct */
  mwSize dim = (mwSize) nmodes;
  mxArray * csf = mxCreateCellArray(1, &dim);

  splatt_idx_t m;
  for(m=0; m < nmodes; ++m) {
    uint64_t * data;
    mxArray * curr =
        mxCreateStructMatrix(1, 1, sizeof(keys)/sizeof(keys[0]), keys);

    /* fill in each cell */
    __mk_uint64(curr, "nnz", 1, &(tt[m]->nnz));
    __mk_uint64(curr, "nmodes", 1, &(nmodes));
    __mk_uint64(curr, "dims", nmodes, tt[m]->dims);
    __mk_uint64(curr, "dim_perm", nmodes, tt[m]->dim_perm);
    __mk_uint64(curr, "nslcs", 1, &(tt[m]->nslcs));
    __mk_uint64(curr, "nfibs", 1, &(tt[m]->nfibs));
    __mk_uint64(curr, "sptr", tt[m]->nslcs+1, tt[m]->sptr);
    __mk_uint64(curr, "fptr", tt[m]->nfibs+1, tt[m]->fptr);
    __mk_uint64(curr, "fids", tt[m]->nfibs, tt[m]->fids);
    __mk_uint64(curr, "inds", tt[m]->nnz, tt[m]->inds);
    __mk_double(curr, "vals", tt[m]->nnz, tt[m]->vals);

    if(tt[m]->indmap != NULL) {
      __mk_uint64(curr, "indmap", tt[m]->nslcs, tt[m]->indmap);
    } else {
      uint64_t no = 0;
      __mk_uint64(curr, "has_indmap", 1, &(no));
    }

    /* tiled fields */
    __mk_int32(curr, "tiled", 1, &(tt[m]->tiled));
    if(tt[m]->tiled != SPLATT_NOTILE) {
      __mk_uint64(curr, "nslabs", 1, &(tt[m]->nslabs));
      __mk_uint64(curr, "slabptr", tt[m]->nslabs+1, tt[m]->slabptr);
      __mk_uint64(curr, "sids", tt[m]->nfibs, tt[m]->sids);
    }

    /* store struct */
    mxSetCell(csf, m, curr);
  }

  return csf;
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
  if(nrhs == 0) {
    mexErrMsgTxt("ARG1 must be a file or sptensor\n");
  }

  double * cpd_opts = splatt_default_opts();

  splatt_idx_t nmodes;
  splatt_csf_t ** tt = NULL;
  /* parse the tensor from a file */
  if(mxIsChar(prhs[0])) {
    char * fname = (char *) mxArrayToString(prhs[0]);
    tt = splatt_csf_load(fname, &nmodes, cpd_opts);
    mxFree(fname);
  } else if(nrhs == 2) {
    tt = __convert_sptensor(prhs[0], prhs[1], &nmodes, cpd_opts);
  } else {
    mexErrMsgTxt("Missing arguments. See 'help splatt_load' for usage.\n");
    return;
  }

  mxArray * csf = __pack_csf(tt, nmodes);

  if(nlhs > 0) {
    plhs[0] = csf;
  }
}

