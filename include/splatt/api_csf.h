#ifndef SPLATT_SPLATT_CSF_H
#define SPLATT_SPLATT_CSF_H


/*
 * COMPRESSED SPARSE FIBER API
 */


#ifdef __cplusplus
extern "C" {
#endif


/**
\defgroup api_csf_list List of functions for \splatt CSF tensors.
@{
*/

/**
* @brief Read a tensor from a file and convert to CSF format.
*
* @param fname The filename to read from.
* @param[out] nmodes SPLATT will fill in the number of modes found.
* @param[out] tensors An array of splatt_csf structure(s). Allocation scheme
*                follows opts[SPLATT_OPTION_CSF_ALLOC].
* @param options An options array allocated by splatt_default_opts().
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_csf_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_csf ** tensors,
    double const * const options);


/**
* @brief Convert a tensor in coordinate format [(i,j,k]=v] to CSF.
*
* @param nmodes The number of modes in the tensor.
* @param nnz The number of nonzero values in the tensor.
* @param inds An array of indices for each mode. Nonzero 'n' is found at
*             inds[0][n-1], inds[1][n-1], ..., inds[nmodes-1][n-1].
* @param vals The actual values of the nonzeros. Nonzero 'n' is found at
*             vals[n-1].
* @param[out] tensors An array of splatt_csf structure(s). Allocation scheme
*                follows opts[SPLATT_OPTION_CSF_ALLOC].
* @param options Options array allocated by splatt_default_opts(). Use the
*                splatt_option_t enum to change these values.
*                SPLATT_OPTION_TILE is used here.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_csf_convert(
    splatt_idx_t const nmodes,
    splatt_idx_t const nnz,
    splatt_idx_t ** const inds,
    splatt_val_t * const vals,
    splatt_csf ** tensors,
    double const * const options);


/**
* @brief Free all memory allocated for a tensor in CSF form.
*
* @param csf The tensor(s) to free.
* @param opts opts[SPLATT_OPTION_CSF_ALLOC] tells us how many tensors are
*             allocated.
*/
void splatt_free_csf(
    splatt_csf * tensors,
    double const * const options);



/** @} */


#ifdef __cplusplus
}
#endif

#endif
