#ifndef SPLATTLAB_SHARED_H
#define SPLATTLAB_SHARED_H

#include <splatt.h>
#include <stdio.h>



/******************************************************************************
 * STRUCTURE ACCESS
 *****************************************************************************/

/**
* @brief Extract a uint64_t pointer from a Matlab struct, given a string
*        fieldname.
*
* @param mxstruct The Matlab struct.
* @param field The field to extract.
*
* @return A pointer to mxstruct.fieldname.
*/
static uint64_t * p_get_uint64_data(
    mxArray const * const mxstruct,
    char const * const field)
{
  return (uint64_t *) mxGetData(mxGetField(mxstruct, 0, field));
}

/**
* @brief Extract a double pointer from a Matlab struct, given a string
*        fieldname.
*
* @param mxstruct The Matlab struct.
* @param field The field to extract.
*
* @return A pointer to mxstruct.fieldname.
*/
static double * p_get_double_data(
    mxArray const * const mxstruct,
    char const * const field)
{
  return (double *) mxGetData(mxGetField(mxstruct, 0, field));
}


/**
* @brief Extract a int pointer from a Matlab struct, given a string
*        fieldname.
*
* @param mxstruct The Matlab struct.
* @param field The field to extract.
*
* @return A pointer to mxstruct.fieldname.
*/
static int * p_get_int_data(
    mxArray const * const mxstruct,
    char const * const field)
{
  return (int *) mxGetData(mxGetField(mxstruct, 0, field));
}


/**
* @brief Create a struct field of type int32_t.
*
* @param mxstruct The struct to add a field to.
* @param field The name of the field to add.
* @param len The number of elements.
* @param vals The values to copy in.
*/
static void p_mk_int32(
    mxArray * const mxstruct,
    char const * const field,
    splatt_idx_t const len,
    int32_t const * const vals)
{
  mxSetField(mxstruct, 0, field,
      mxCreateNumericMatrix(1, len, mxINT32_CLASS, mxREAL));
  memcpy(p_get_uint64_data(mxstruct, field), vals, len * sizeof(int32_t));
}


/**
* @brief Create a struct field of type int64_t.
*
* @param mxstruct The struct to add a field to.
* @param field The name of the field to add.
* @param len The number of elements.
* @param vals The values to copy in.
*/
static void p_mk_uint64(
    mxArray * const mxstruct,
    char const * const field,
    splatt_idx_t const len,
    splatt_idx_t const * const vals)
{
  mxSetField(mxstruct, 0, field,
      mxCreateNumericMatrix(1, len, mxUINT64_CLASS, mxREAL));
  memcpy(p_get_uint64_data(mxstruct, field), vals, len * sizeof(uint64_t));
}


/**
* @brief Create a struct field of type double.
*
* @param mxstruct The struct to add a field to.
* @param field The name of the field to add.
* @param len The number of elements.
* @param vals The values to copy in.
*/
static void p_mk_double(
    mxArray * const mxstruct,
    char const * const field,
    splatt_idx_t const len,
    splatt_val_t const * const vals)
{
  mxSetField(mxstruct, 0, field,
      mxCreateDoubleMatrix(1, len, mxREAL));
  memcpy(p_get_uint64_data(mxstruct, field), vals, len * sizeof(double));
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


static void p_parse_opts(
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
* @return An array of splatt_csf tensors.
*/
static splatt_csf * p_convert_sptensor(
    mxArray const * const mat_inds,
    mxArray const * const mat_vals,
    splatt_idx_t * const nmodes,
    double const * const cpd_opts)
{
  splatt_idx_t m;

  /* parse from tensor toolbox's sptensor */
  mwSize const * dims = mxGetDimensions(mat_inds);
  splatt_idx_t nnz = dims[0];
  *nmodes = dims[1];

  /* allocate extra tensor for re-arranging */
  splatt_val_t * vals = (splatt_val_t *) mxMalloc(nnz * sizeof(splatt_val_t));
  splatt_idx_t * inds[SPLATT_MAX_NMODES];
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

  splatt_csf * tt;
  splatt_csf_convert(*nmodes, nnz, inds, vals, &tt, cpd_opts);

  for(m=0; m < *nmodes; ++m) {
    mxFree(inds[m]);
  }
  mxFree(vals);

  return tt;
}


/**
* @brief Convert an mxArray to splatt_csf.
*
* @param cell The cell array to convert.
* @param[out] outnmodes The number of modes in the tensor.
*
* @return The CSF tensor, unpacked.
*/
static splatt_csf * p_unpack_csf_cell(
    mxArray const * const cell,
    splatt_idx_t * outnmodes)
{
  splatt_idx_t t, i, tile;
  splatt_csf * csf = NULL;
  splatt_idx_t ntensors = (splatt_idx_t) mxGetNumberOfElements(cell);

  csf = (splatt_csf *) mxMalloc(ntensors * sizeof(*csf));

  for(t=0; t < ntensors; ++t) {
    mxArray const * const curr = mxGetCell(cell, t);

    csf[t].nnz = *(p_get_uint64_data(curr, "nnz"));
    csf[t].nmodes = *(p_get_uint64_data(curr, "nmodes"));

    splatt_idx_t const nmodes = csf[t].nmodes;

    memcpy(csf[t].dims, p_get_uint64_data(curr, "dims"),
        nmodes * sizeof(uint64_t));
    memcpy(csf[t].dim_perm, p_get_uint64_data(curr, "dim_perm"),
        nmodes * sizeof(uint64_t));
    memcpy(&(csf[t].which_tile), p_get_int_data(curr, "which_tile"),
        sizeof(splatt_tile_type));
    memcpy(&(csf[t].ntiles), p_get_uint64_data(curr, "ntiles"),
        sizeof(uint64_t));
    memcpy(csf[t].tile_dims, p_get_uint64_data(curr, "tile_dims"),
        nmodes * sizeof(uint64_t));

    /* allocate sparsity patterns */
    csf[t].pt = (csf_sparsity *)mxMalloc(csf[t].ntiles * sizeof(csf_sparsity));

    /* extract each tile */
    mxArray const * const pts = mxGetField(curr, 0, "pt");
    for(tile=0; tile < csf[t].ntiles; ++tile) {
      csf_sparsity * pt = csf[t].pt + tile;
      mxArray const * const curr_tile = mxGetCell(pts, tile);

      memcpy(pt->nfibs, p_get_uint64_data(curr_tile, "nfibs"),
          nmodes * sizeof(uint64_t));
      /* check for empty tile */
      if(pt->nfibs[nmodes-1] == 0) {
        pt->vals = NULL;

        splatt_idx_t m;
        for(m=0; m < nmodes; ++m) {
          pt->fptr[m] = NULL;
          pt->fids[m] = NULL;
        }
        /* first fptr may be accessed anyway */
        pt->fptr[0] = (splatt_idx_t *) malloc(2 * sizeof(**(pt->fptr)));
        pt->fptr[0][0] = 0;
        pt->fptr[0][1] = 0;

        continue;
      }

      pt->vals = p_get_double_data(curr_tile, "vals");

      /* figure out which fids[*] exist */
      int32_t has_fids[SPLATT_MAX_NMODES];
      memcpy(has_fids, mxGetData(mxGetField(curr_tile, 0, "has_fids")),
          nmodes * sizeof(int32_t));

      /* grab fptr/fids */
      mxArray const * const mxfptr = mxGetField(curr_tile, 0, "fptr");
      splatt_idx_t m;
      for(m=0; m < nmodes-1; ++m) {
        pt->fptr[m] = mxGetData(mxGetCell(mxfptr, m));
      }

      mxArray const * const mxfids = mxGetField(curr_tile, 0, "fids");
      for(m=0; m < nmodes; ++m) {
        if(has_fids[m]) {
          pt->fids[m] = mxGetData(mxGetCell(mxfids, m));
        } else {
          pt->fids[m] = NULL;
        }
      }
    } /* foreach tile */
  } /* foreach tensor */

  *outnmodes = csf->nmodes;
  return csf;
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
static splatt_csf * p_parse_tensor(
    int const nargs,
    mxArray const * const args[],
    splatt_idx_t * nmodes,
    double const * const cpd_opts)
{
  splatt_csf * tt = NULL;
  if(nargs < 1) {
    mexErrMsgTxt("Missing arguments. See 'help splatt_load' for usage.\n");
    return NULL;
  }

  if(mxIsChar(args[0])) {
    char * fname = (char *) mxArrayToString(args[0]);
    splatt_csf_load(fname, nmodes, &tt, cpd_opts);
    mxFree(fname);
  } else if(nargs > 1 && mxIsNumeric(args[0]) && mxIsNumeric(args[1])) {
    tt = p_convert_sptensor(args[0], args[1], nmodes, cpd_opts);
  } else if(mxIsCell(args[0])) {
    tt = p_unpack_csf_cell(args[0], nmodes);
  } else {
    mexErrMsgTxt("Invalid tensor format. See 'help splatt_load' for usage.\n");
    return NULL;
  }

  return tt;
}


/**
* @brief Free the memory allocated from a CSF tensor. The process may change
*        depending on the data source.
*
* @param nargs The number of arguments (total, may include other params)
* @param args[] The arguments we can access.
* @param nmodes The number of modes in the tensor.
* @param tt The tensor to free.
*/
static void p_free_tensor(
    int const nargs,
    mxArray const * const args[],
    splatt_csf * tt,
    double const * const splatt_opts)
{
  if(mxIsChar(args[0])) {
    splatt_free_csf(tt, splatt_opts);
  } else if(nargs > 1 && mxIsNumeric(args[0]) && mxIsNumeric(args[1])) {
    splatt_free_csf(tt, splatt_opts);
  } else if(mxIsCell(args[0])) {
    /* pointer is mxMalloc'ed, we don't have to do anything */
  } else {
    mexErrMsgTxt("Invalid tensor format. See 'help splatt_load' for usage.\n");
  }
}


#endif
