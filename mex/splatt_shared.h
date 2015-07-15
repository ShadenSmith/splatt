#ifndef SPLATTLAB_SHARED_H
#define SPLATTLAB_SHARED_H

#include <splatt.h>



/******************************************************************************
 * STRUCTURE ACCESS
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

static void __mk_int32(
    mxArray * const mxstruct,
    char const * const field,
    splatt_idx_t const len,
    int32_t const * const vals)
{
  mxSetField(mxstruct, 0, field,
      mxCreateNumericMatrix(1, len, mxINT32_CLASS, mxREAL));
  memcpy(__get_uint64_data(mxstruct, field), vals, len * sizeof(int32_t));
}

static void __mk_uint64(
    mxArray * const mxstruct,
    char const * const field,
    splatt_idx_t const len,
    splatt_idx_t const * const vals)
{
  mxSetField(mxstruct, 0, field,
      mxCreateNumericMatrix(1, len, mxUINT64_CLASS, mxREAL));
  memcpy(__get_uint64_data(mxstruct, field), vals, len * sizeof(uint64_t));
}

static void __mk_double(
    mxArray * const mxstruct,
    char const * const field,
    splatt_idx_t const len,
    splatt_val_t const * const vals)
{
  mxSetField(mxstruct, 0, field,
      mxCreateDoubleMatrix(1, len, mxREAL));
  memcpy(__get_uint64_data(mxstruct, field), vals, len * sizeof(double));
}

/******************************************************************************
 * OPTION PARSING
 *****************************************************************************/
typedef struct splatt_option
{
  char * name;
  int has_arg;
  int opt_id;
} splattlab_option_t;

static splattlab_option_t option_names[] =
{
  {"tol", 1, SPLATT_OPTION_TOLERANCE},
  {"its", 1, SPLATT_OPTION_NITER},
  {"threads", 1, SPLATT_OPTION_NTHREADS},
  {"verbosity", 1, SPLATT_OPTION_VERBOSITY},
  {NULL, 0, 0}
};


static void __parse_opts(
    mxArray const * const opts,
    double * const cpd_opts)
{
  if(!mxIsStruct(opts)) {
    mexErrMsgTxt("SPLATT expects options array to be a structure.\n");
    return;
  }

  splattlab_option_t * head = option_names;
  for(; head->name != NULL; ++head) {
    mxArray * mopt = mxGetField(opts, 0, head->name);
    if(!mopt) {
      continue;
    }

    if(head->has_arg == 1) {
      cpd_opts[head->opt_id] = (double) mxGetScalar(mopt);
    } else {
      cpd_opts[head->opt_id] = 1;
    }
  }
}



/******************************************************************************
 * IO FUNCTIONS
 *****************************************************************************/

/**
* @brief Convert an inds matrix and vals vector (from sptensor) to CSF.
*
* @param mat_inds An (nnz x nmodes) matrix of indices.
* @param mat_vals An (nnz x 1) vector of values.
* @param nmodes A point to to be set specifying the number of found modes.
* @param cpd_opts SPLATT options array.
*
* @return An array of splatt_csf_t tensors.
*/
static splatt_csf_t * __convert_sptensor(
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
  splatt_val_t * vals = (splatt_val_t *) mxMalloc(nnz * sizeof(splatt_val_t));
  splatt_idx_t * inds[MAX_NMODES];
  for(m=0; m < *nmodes; ++m) {
    inds[m] = (splatt_idx_t *) mxMalloc(nnz * sizeof(splatt_idx_t));
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

  splatt_csf_t * tt;
  splatt_csf_convert(*nmodes, nnz, inds, vals, &tt, cpd_opts);

  for(m=0; m < *nmodes; ++m) {
    mxFree(inds[m]);
  }
  mxFree(vals);

  return tt;
}


static splatt_csf_t * __unpack_csf_cell(
    mxArray const * const cell,
    splatt_idx_t * outnmodes)
{
  splatt_idx_t m;
  splatt_csf_t * tt = NULL;
  splatt_idx_t nmodes = (splatt_idx_t) mxGetNumberOfElements(cell);

  tt = (splatt_csf_t *) mxMalloc(nmodes * sizeof(splatt_csf_t));

  for(m=0; m < nmodes; ++m) {
    mxArray const * const curr = mxGetCell(cell, m);

    tt[m].nmodes = nmodes;
    tt[m].nnz = *(__get_uint64_data(curr, "nnz"));
    memcpy(tt[m].dims, __get_uint64_data(curr, "dims"),
        nmodes * sizeof(uint64_t));
    memcpy(tt[m].dim_perm, __get_uint64_data(curr, "dim_perm"),
        nmodes * sizeof(uint64_t));
    tt[m].nslcs = *(__get_uint64_data(curr, "nslcs"));
    tt[m].nfibs = *(__get_uint64_data(curr, "nfibs"));
    tt[m].sptr = __get_uint64_data(curr, "sptr");
    tt[m].fptr = __get_uint64_data(curr, "fptr");
    tt[m].fids = __get_uint64_data(curr, "fids");
    tt[m].inds = __get_uint64_data(curr, "inds");
    tt[m].vals = __get_double_data(curr, "vals");

    if(*__get_uint64_data(curr, "has_indmap") == 1) {
      tt[m].indmap = __get_uint64_data(curr, "indmap");
    }

    tt[m].tiled = (int) *(__get_uint64_data(curr, "tiled"));
    if(tt[m].tiled != SPLATT_NOTILE) {
      tt[m].nslabs = *(__get_uint64_data(curr, "nslabs"));
      tt[m].slabptr = __get_uint64_data(curr, "slabptr");
      tt[m].sids = __get_uint64_data(curr, "sids");
    }
  }

  *outnmodes = nmodes;
  return tt;
}


/**
* @brief Parse a CSF tensor from a number of argument formats:
*     1. Filename     - first arg must be a character string.
*     2. (inds, vals) - first two args must be matrices.
*     3. Cell array   - right now this just assumes a cell array of already
*                       existing CSF. TODO: can this be better?
*     4. TODO: handle TT's sptensor class too (does not seem to be possible in
*              Octave)
*
* @param nargs The number of arguments (total, may include other params)
* @param args[] The arguments we can access.
* @param nmodes A pointer which we must fill with the number of modes found.
* @param cpd_opts SPLATT options array.
*
* @return A list of CSF tensors.
*/
static splatt_csf_t * __parse_tensor(
    int const nargs,
    mxArray const * const args[],
    splatt_idx_t * nmodes,
    double const * const cpd_opts)
{
  splatt_csf_t * tt = NULL;
  if(nargs < 1) {
    mexErrMsgTxt("Missing arguments. See 'help splatt_load' for usage.\n");
    return NULL;
  }

  if(mxIsChar(args[0])) {
    char * fname = (char *) mxArrayToString(args[0]);
    splatt_csf_load(fname, nmodes, &tt, cpd_opts);
    mxFree(fname);
  } else if(nargs > 1 && mxIsNumeric(args[0]) && mxIsNumeric(args[1])) {
    tt = __convert_sptensor(args[0], args[1], nmodes, cpd_opts);
  } else if(mxIsCell(args[0])) {
    tt = __unpack_csf_cell(args[0], nmodes);
  } else {
    mexErrMsgTxt("Invalid tensor format. See 'help splatt_load' for usage.\n");
    return NULL;
  }

  return tt;
}


#endif
