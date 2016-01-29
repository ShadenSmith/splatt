#ifndef SPLATT_SPLATT_H
#define SPLATT_SPLATT_H




/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <inttypes.h>
#include <float.h>

#ifdef SPLATT_USE_MPI
#include <mpi.h>
#endif





/******************************************************************************
 * TYPES
 *****************************************************************************/

/* USER: You may edit values to chance the size of integer and real types.
 * Accepted values are 32 and 64. Changing these values to 32 will decrease
 * memory consumption at the cost of precision and maximum supported tensor
 * size. */

#ifndef SPLATT_IDX_TYPEWIDTH
  #define SPLATT_IDX_TYPEWIDTH 32
#endif

#ifndef SPLATT_VAL_TYPEWIDTH
  #define SPLATT_VAL_TYPEWIDTH 64
#endif


/* Set type constants based on width. */
#if   SPLATT_IDX_TYPEWIDTH == 32
  typedef uint32_t splatt_idx_t;
  #define SPLATT_IDX_MAX UINT32_MAX
  #define SPLATT_PF_IDX PRIu32
  #define SPLATT_MPI_IDX MPI_UINT32_T

#elif SPLATT_IDX_TYPEWIDTH == 64
  typedef uint64_t splatt_idx_t;
  #define SPLATT_IDX_MAX UINT64_MAX
  #define SPLATT_PF_IDX PRIu64
  #define SPLATT_MPI_IDX MPI_UINT64_T
#else
  #error *** Incorrect user-supplied value of SPLATT_IDX_TYPEWIDTH ***
#endif


#if   SPLATT_VAL_TYPEWIDTH == 32
  typedef float splatt_val_t;
  #define SPLATT_VAL_MIN FLT_MIN
  #define SPLATT_VAL_MAX FLT_MAX
  #define SPLATT_PF_VAL "f"
  #define SPLATT_MPI_VAL MPI_FLOAT

#elif SPLATT_VAL_TYPEWIDTH == 64
  typedef double splatt_val_t;
  #define SPLATT_VAL_MIN DBL_MIN
  #define SPLATT_VAL_MAX DBL_MAX
  #define SPLATT_PF_VAL "f"
  #define SPLATT_MPI_VAL MPI_DOUBLE

#else
  #error *** Incorrect user-supplied value of SPLATT_VAL_TYPEWIDTH ***
#endif


/******************************************************************************
 * VERSION
 *****************************************************************************/
#define SPLATT_VER_MAJOR     1
#define SPLATT_VER_MINOR     1
#define SPLATT_VER_SUBMINOR  1



/******************************************************************************
 * ENUMS / CONSTANTS
 *****************************************************************************/

#ifndef SPLATT_MAX_NMODES
#define SPLATT_MAX_NMODES ((splatt_idx_t) 8)
#endif

/**
* @brief Enum for defining SPLATT options. Use the splatt_default_opts() and
*        splatt_free_opts() functions to initialize and free an options array.
*/
typedef enum
{
  /* high level options */
  SPLATT_OPTION_NTHREADS,   /* Number of OpenMP threads to use. */
  SPLATT_OPTION_TOLERANCE,  /* Threshold for convergence. */
  SPLATT_OPTION_NITER,      /* Maximum number of iterations to perform. */
  SPLATT_OPTION_VERBOSITY,  /* Verbosity level */

  /* low level options */
  SPLATT_OPTION_RANDSEED,   /* Random number seed */
  SPLATT_OPTION_CSF_ALLOC,  /* How many (and which) tensors to allocate. */
  SPLATT_OPTION_TILE,       /* Use cache tiling during MTTKRP. */
  SPLATT_OPTION_TILEDEPTH,  /* Minimium depth in CSF to tile, 0-indexed. */

  SPLATT_OPTION_DECOMP,     /* Decomposition to use on distributed systems */
  SPLATT_OPTION_COMM,       /* Communication pattern to use */

  SPLATT_OPTION_NOPTIONS    /* Gives the size of the options array. */
} splatt_option_type;


/**
* @brief Return codes used by SPLATT.
*/
typedef enum
{
  SPLATT_SUCCESS = 1,     /* Successful SPLATT API call. */
  SPLATT_ERROR_BADINPUT,  /* SPLATT found an issue with the input.
                             Try splatt-check to debug. */
  SPLATT_ERROR_NOMEMORY   /* SPLATT did not have enough memory to complete.
                             Try using fewer factors, a smaller tensor, or
                             recompile with less precision. */
} splatt_error_type;


/**
* @brief Verbosity levels used by SPLATT.
*/
typedef enum
{
  SPLATT_VERBOSITY_NONE, /* Nothing written to STDOUT. */
  SPLATT_VERBOSITY_LOW,  /* Only headers/footers and high-level timing. */
  SPLATT_VERBOSITY_HIGH, /* Timers for all modes. */
  SPLATT_VERBOSITY_MAX   /* All output, including detailed timers. */
} splatt_verbosity_type;


/**
* @brief Types of tiling used by SPLATT.
*/
typedef enum
{
  SPLATT_NOTILE,
  SPLATT_DENSETILE,
  /* DEPRECATED - pending CSF implementations */
  SPLATT_SYNCTILE,
  SPLATT_COOPTILE,
} splatt_tile_type;


/**
* @brief Types of CSF allocation available.
*/
typedef enum
{
  SPLATT_CSF_ONEMODE, /** Only allocate one CSF for factorization. */
  SPLATT_CSF_TWOMODE, /** Allocate one for the smallest and largest modes. */
  SPLATT_CSF_ALLMODE, /** Allocate one CSF for every mode. */
} splatt_csf_type;


/**
* @brief Tensor decomposition schemes.
*/
typedef enum
{
  /** @brief Coarse-grained decomposition is using a separate 1D decomposition
   *         for each mode. */
  SPLATT_DECOMP_COARSE,
  /** @brief Medium-grained decomposition is an 'nmodes'-dimensional
   *         decomposition. */
  SPLATT_DECOMP_MEDIUM,
  /** @brief Fine-grained decomposition distributes work at the nonzero level.
   *         NOTE: requires a partitioning on the nonzeros. */
  SPLATT_DECOMP_FINE
} splatt_decomp_type;


/**
* @brief Communication pattern type. We support point-to-point, and all-to-all
*        (vectorized).
*/
typedef enum
{
  SPLATT_COMM_POINT2POINT,
  SPLATT_COMM_ALL2ALL
} splatt_comm_type;


static double const SPLATT_VAL_OFF = -DBL_MAX;




/******************************************************************************
 * DATA STRUCTURES
 *****************************************************************************/

/**
* @brief Kruskal tensors are the output of the CPD. Each mode of the tensor is
*        represented as a matrix with unit columns. Lambda is a vector whose
*        entries scale the columns of the matrix factors.
*/
typedef struct splatt_kruskal
{
  /** @brief The rank of the decomposition. */
  splatt_idx_t rank;

  /** @brief The row-major matrix factors for each mode. */
  splatt_val_t * factors[SPLATT_MAX_NMODES];

  /** @brief Length-for each column. */
  splatt_val_t * lambda;

  /** @brief The number of modes in the tensor. */
  splatt_idx_t nmodes;

  /** @brief The number of rows in each factor. */
  splatt_idx_t dims[SPLATT_MAX_NMODES];

  /** @brief The quality [0,1] of the CPD */
  double fit;
} splatt_kruskal;



/**
* @brief The sparsity pattern of a CSF (sub-)tensor.
*/
typedef struct
{
  /** @brief The size of each fptr and fids array. */
  splatt_idx_t nfibs[SPLATT_MAX_NMODES];

  /** @brief The pointer structure for each sub-tree. fptr[f] marks the start
   *         of the children of node 'f'. This structure is a generalization of
   *         the 'rowptr' array used for CSR matrices. */
  splatt_idx_t * fptr[SPLATT_MAX_NMODES];

  /** @brief The index of each node. These map nodes back to the original
   *         tensor nonzeros. */
  splatt_idx_t * fids[SPLATT_MAX_NMODES];

  /** @brief The actual nonzero values. This array is of length
   *         nfibs[nmodes-1]. */
  splatt_val_t * vals;
} csf_sparsity;



/**
* @brief CSF tensors are the compressed storage format for performing fast
*        tensor computations in the SPLATT library.
*/
typedef struct splatt_csf
{
  /** @brief The number of nonzeros. */
  splatt_idx_t nnz;

  /** @brief The number of modes. */
  splatt_idx_t nmodes;

  /** @brief The dimension of each mode. */
  splatt_idx_t dims[SPLATT_MAX_NMODES];

  /** @brief The permutation of the tensor modes.
   *         dim_perm[0] is the mode stored at the root level and so on. */
  splatt_idx_t dim_perm[SPLATT_MAX_NMODES];

  /** @brief Which tiling scheme this tensor is stored as. */
  splatt_tile_type which_tile;

  /** @brief How many tiles there are. */
  splatt_idx_t ntiles;

  /** @brief For a dense tiling, how many tiles along each mode. */
  splatt_idx_t tile_dims[SPLATT_MAX_NMODES];

  /** @brief Sparsity structures -- one for each tile. */
  csf_sparsity * pt;
} splatt_csf;





/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/*
 * OPTIONS API
 */

/**
\defgroup api_opt_list List of functions for \splatt options.
@{
*/


/**
* @brief Allocate and fill an options array with default options.
*
* @return The options array.
*/
double * splatt_default_opts(void);


/**
* @brief Free an options array allocated with splatt_default_opts().
*/
void  splatt_free_opts(
    double * opts);

/** @} */


/*
 * DATA STRUCTURE API
 */

/**
\defgroup api_struct_list List of functions for \splatt data structures.
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


/**
* @brief Free a splatt_kruskal allocated by splatt_cpd().
*
* @param factored The factored tensor to free.
*/
void splatt_free_kruskal(
    splatt_kruskal * factored);

/** @} */

/*
 * FACTORIZATION API
 */

/**
\defgroup api_fact_list List of functions for tensor factorizations.
@{
*/


/**
* @brief Compute the CPD using alternating least squares.
*
* @param nfactors The rank of the decomposition to perform.
* @param nmodes The number of modes in the tensor. Optimizations are currently
*               only present for nmodes=3.
* @param tensors An array of splatt_csf created by SPLATT.
* @param options Options array for SPLATT.
* @param[out] factored The factored tensor in Kruskal format.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_cpd_als(
    splatt_csf const * const tensors,
    splatt_idx_t const nfactors,
    double const * const options,
    splatt_kruskal * factored);


/**
* @brief Compute the estimated value of a coordinate, given a factored tensor.
*        This is equal to:
*             \sum_{r=1}^{rank} [A(i,r) * B(j,r) * C(k,r) ... ]
*
* @param factored The factored tensor.
* @param coords The index to estimate.
*
* @return The estimated value.
*/
splatt_val_t splatt_kruskal_estimate(
    splatt_kruskal const * const factored,
    splatt_idx_t const * const coords);


/** @} */




/**
\defgroup api_op_list List of functions for tensor operations.
@{
*/

/**
* @brief Matricized Tensor times Khatri-Rao Product (MTTKRP) with a sparse
*        tensor in CSF format.
*
* @param mode Which mode we are operating on.
* @param ncolumns How many columns each matrix has ('nfactors').
* @param tensor The CSF tensor to multipy with.
* @param matrices The row-major dense matrices to multiply with.
* @param[out] matout The output matrix.
* @param options SPLATT options array.
*
* @return SPLATT error code. SPLATT_SUCCESS on success.
*/
int splatt_mttkrp(
    splatt_idx_t const mode,
    splatt_idx_t const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const matout,
    double const * const options);

/** @} */

/*
 * MPI API
 */


#ifdef SPLATT_USE_MPI
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
#endif


#ifdef __cplusplus
}
#endif


#endif
