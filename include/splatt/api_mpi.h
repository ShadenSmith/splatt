#ifndef SPLATT_SPLATT_MPI_H
#define SPLATT_SPLATT_MPI_H


#ifdef SPLATT_USE_MPI

/*
 * MPI API
 */


#ifdef __cplusplus
extern "C" {
#endif


/**
\defgroup api_mpi_list List of functions for \splatt MPI.
@{
*/
/*
 * TODO: There is currently no MPI support for factorization. That is due in
 *       version 1.2.x.
 */

/**
* @brief Read a tensor from a file, distribute among an MPI communicator, and
*        convert to CSF format.
*
* @param fname The filename to read from.
* @param[out] nmodes SPLATT will fill in the number of modes found.
* @param[out] tensors An array of splatt_csf structure(s). Allocation scheme
*                follows opts[SPLATT_OPTION_CSF_ALLOC].
* @param options An options array allocated by splatt_default_opts(). The
*                distribution scheme follows opts[SPLATT_OPTION_DECOMP].
* @param comm The MPI communicator to distribute among.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_mpi_csf_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_csf ** tensors,
    double const * const options,
    MPI_Comm comm);



/**
* @brief Load a tensor in coordinate from from a file and distribute it among
*        an MPI communicator.
*
* @param fname The file to read from.
* @param[out] nmodes The number of modes in the tensor.
* @param[out] nnz The number of nonzeros in my portion.
* @param[out] inds An array of indices for each mode.
* @param[out] vals The tensor nonzero values.
* @param options SPLATT options array. Currently unused.
* @param comm Which communicator to distribute among.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_mpi_coord_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_idx_t * nnz,
    splatt_idx_t *** inds,
    splatt_val_t ** vals,
    double const * const options,
    MPI_Comm comm);




#ifdef __cplusplus
}
#endif


/** @} */


#endif /* if mpi */

#endif
