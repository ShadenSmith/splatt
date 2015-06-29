
#include "mex.h"

#include <stdint.h>
#include <string.h>
#include <splatt.h>


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


/******************************************************************************
 * ENTRY FUNCTION
 *****************************************************************************/
void mexFunction(
    int nlhs,
    mxArray * plhs[],
    int nrhs,
    mxArray const * prhs[])
{
  splatt_idx_t m, m2;

  if(nrhs != 1 || !mxIsChar(prhs[0])) {
    mexErrMsgTxt("ARG1 must be a file\n");
  }

  /* parse the tensor */
  char * fname = (char *) mxArrayToString(prhs[0]);
  double * cpd_opts = splatt_default_opts();
  splatt_idx_t nmodes;
  splatt_csf_t ** tt = splatt_csf_load(fname, &nmodes, cpd_opts);
  mxFree(fname);

  /* create splatt_csf_t matlab struct */
  mwSize dim = (mwSize) nmodes;
  mxArray * csf = mxCreateCellArray(1, &dim);

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

  if(nlhs > 0) {
    plhs[0] = csf;
  }
}

