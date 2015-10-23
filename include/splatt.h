#ifndef SPLATT_SPLATT_H
#define SPLATT_SPLATT_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <inttypes.h>
#include <float.h>


/******************************************************************************
 * TYPES
 *****************************************************************************/

/* USER: You may edit values to chance the size of integer and real types.
 * Accepted values are 32 and 64. Changing these values to 32 will decrease
 * memory consumption at the cost of precision and maximum supported tensor
 * size. */
#define SPLATT_IDX_TYPEWIDTH 64
#define SPLATT_VAL_TYPEWIDTH 64


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
  #error "Incorrect user-supplied value of SPLATT_IDX_TYPEWIDTH"
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
  #error "Incorrect user-supplied value fo SPLATT_VAL_TYPEWIDTH"
#endif


/******************************************************************************
 * VERSION
 *****************************************************************************/
#define SPLATT_VER_MAJOR     0
#define SPLATT_VER_MINOR     0
#define SPLATT_VER_SUBMINOR  1



/******************************************************************************
 * ENUMS / CONSTANTS
 *****************************************************************************/

#define MAX_NMODES ((splatt_idx_t) 8)

/**
* @brief Enum for defining SPLATT options. Use the splatt_default_opts() and
*        splatt_free_opts() functions to initialize and free an options array.
*/
typedef enum
{
  SPLATT_OPTION_TOLERANCE,  /** Threshold for convergence. */
  SPLATT_OPTION_NITER,      /** Maximum number of iterations to perform. */
  SPLATT_OPTION_CSF_ALLOC,  /** How many (and which) tensors to allocate. */
  SPLATT_OPTION_TILE,       /** Use cache tiling during MTTKRP. */
  SPLATT_OPTION_NTHREADS,   /** Number of OpenMP threads to use. */
  SPLATT_OPTION_RANDSEED,   /** Random number seed */
  SPLATT_OPTION_VERBOSITY,  /** Verbosity level */
  SPLATT_OPTION_NOPTIONS    /** Gives the size of the options array. */
} splatt_option_t;

/**
* @brief Return codes used by SPLATT.
*/
typedef enum
{
  SPLATT_SUCCESS = 1,     /** Successful SPLATT API call. */
  SPLATT_ERROR_BADINPUT,  /** SPLATT found an issue with the input.
                              Try splatt-check to debug. */
  SPLATT_ERROR_NOMEMORY   /** SPLATT did not have enough memory to complete.
                              Try using fewer factors, a smaller tensor, or
                              recompile with less precision. */
} splatt_error_t;


/**
* @brief Verbosity levels used by SPLATT.
*/
typedef enum
{
  SPLATT_VERBOSITY_NONE, /** Nothing written to STDOUT. */
  SPLATT_VERBOSITY_LOW,  /** Only headers/footers and high-level timing. */
  SPLATT_VERBOSITY_HIGH, /** Timers for all modes. */
  SPLATT_VERBOSITY_MAX   /** All output, including detailed timers. */
} splatt_verbosity_t;


/**
* @brief Types of tiling used by SPLATT.
*/
typedef enum
{
  SPLATT_NOTILE,
  SPLATT_SYNCTILE,
  SPLATT_COOPTILE,
  SPLATT_DENSETILE
} splatt_tile_t;


/**
* @brief Types of CSF allocation available.
*/
typedef enum
{
  SPLATT_CSF_ONEMODE, /** Only allocate one CSF for factorization. */
  SPLATT_CSF_TWOMODE, /** Allocate one for the smallest and largest modes. */
  SPLATT_CSF_ALLMODE, /** Allocate one CSF for every mode. */
} splatt_csf_type;


static double const SPLATT_VAL_OFF = -DBL_MAX;



/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

/**
* @brief Struct describing a Kruskal tensor, allocated and returned by
*        splatt_cpd.
*/
typedef struct splatt_kruskal_t
{
  splatt_idx_t nmodes;                  /** Number of modes (i.e., factors[])*/
  splatt_idx_t rank;                    /** Number of columns in each factor */
  double fit;                           /** The quality [0,1] of the CPD */
  splatt_val_t * lambda;                /** Scaling factors for each column */
  splatt_val_t * factors[MAX_NMODES];   /** Row-major matrix for each mode */
} splatt_kruskal_t;


/*********
 */
typedef struct
{
  splatt_idx_t nfibs[MAX_NMODES];
  splatt_idx_t * fptr[MAX_NMODES];
  splatt_idx_t * fids[MAX_NMODES];
  splatt_val_t * vals;
} csf_sparsity;


typedef struct splatt_csf
{
  splatt_idx_t nnz;
  splatt_idx_t nmodes;
  splatt_idx_t dims[MAX_NMODES];
  splatt_idx_t dim_perm[MAX_NMODES];

  splatt_tile_t which_tile;
  splatt_idx_t ntiles;
  splatt_idx_t tile_dims[MAX_NMODES];

  csf_sparsity * pt; /** sparsity structure -- one for each tile */
} splatt_csf;


/*********
 */


/**
* @brief Struct describing SPLATT's compressed sparse fiber (CSF) format. Use
*        the splatt_csf_* functions to allocate, fill, and free this structure.
*/
typedef struct splatt_csf_t
{
  splatt_idx_t nnz;                     /** Number of nonzeros */
  splatt_idx_t nmodes;                  /** Number of modes */
  splatt_idx_t dims[MAX_NMODES];        /** Dimension of each mode */
  splatt_idx_t dim_perm[MAX_NMODES];    /** Permutation of modes */

  splatt_idx_t  nslcs;    /** Number of slices (length of sptr) */
  splatt_idx_t  nfibs;    /** Number of fibers (length of fptr) */
  splatt_idx_t * sptr;    /** Indexes into fptr the start/end of each slice */
  splatt_idx_t * fptr;    /** Indexes into vals the start/end of each fiber */
  splatt_idx_t * fids;    /** ID of each fiber (for mode dim_perm[nmodes-2])*/
  splatt_idx_t * inds;    /** ID of each nnz (for dim_perm[nmodes-1]) */
  splatt_val_t * vals;    /** Floating point value of each nonzero */

  splatt_idx_t * indmap;  /** Maps local to global indices if empty slices */

  /* TILED STRUCTURES */
  int tiled;                /** splatt_tile_t type */
  splatt_idx_t    nslabs;   /** Number of slabs (length of slabptr) */
  splatt_idx_t * slabptr;   /** Indexes into fptr the start/end of each slab */
  splatt_idx_t * sids;      /** ID of each fiber (for dim_perm[0]) */
} splatt_csf_t;


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

#ifdef __cplusplus
extern 'C' {
#endif

/**
* @brief Compute the CPD using alternating least squares.
*
* @param nfactors The rank of the decomposition to perform.
* @param nmodes The number of modes in the tensor. Optimizations are currently
*               only present for nmodes=3.
* @param tensors An array of splatt_csf_t created by SPLATT.
* @param options Options array for SPLATT.
* @param factored [OUT] The factored tensor in Kruskal format.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_cpd(
    splatt_idx_t const nfactors,
    splatt_idx_t const nmodes,
    splatt_csf const * const tensors,
    double const * const options,
    splatt_kruskal_t * factored);



/**
* @brief Matricized Tensor times Khatri-Rao Product (MTTKRP) with a sparse
*        tensor in CSF format.
*
* @param mode Which mode we are operating on.
* @param ncolumns How many columns each matrix has ('nfactors').
* @param tensor The CSF tensor to multipy with.
* @param matrices The row-major dense matrices to multiply with.
* @param matout The output matrix.
* @param options SPLATT options array.
*
* @return SPLATT error code. SPLATT_SUCCESS on success.
*/
int splatt_mttkrp(
    splatt_idx_t const mode,
    splatt_idx_t const ncolumns,
    splatt_csf_t const * const tensor,
    splatt_val_t ** matrices,
    splatt_val_t * const matout,
    double const * const options);


/**
* @brief Read a tensor from a file and convert to CSF format.
*
* @param fname The filename to read from.
* @param nmodes [OUT] SPLATT will fill in the number of modes found.
* @param tensors [OUT] An array of splatt_csf_t structures, one for each mode.
*                The 'nmodes' param will be filled with the length of the
*                array.
* @param options An options array allocated by splatt_default_opts(). Option
*        SPLATT_OPTION_TILE is used here.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_csf_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_csf_t ** tensors,
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
* @param tensors [OUT] An array of splatt_csf_t structures, one for each mode.
*                The 'nmodes' param will be filled with the length of the
*                array.
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
    splatt_csf_t ** tensors,
    double const * const options);


/**
* @brief Allocate and fill an options array with default options.
*
* @return The options array.
*/
double * splatt_default_opts(void);


/**
* @brief Free the memory allocated by an array of CSF tensors.
*
* @param nmodes The number of modes to free.
* @param tensors The array of CSF tensors. The pointer will also be freed!
*/
void splatt_free_csf(
    splatt_idx_t const nmodes,
    splatt_csf_t * tensors);


/**
* @brief Free an options array allocated with splatt_default_opts().
*/
void  splatt_free_opts(
    double * opts);


/**
* @brief Free a splatt_kruskal_t allocated by splatt_cpd().
*
* @param factored The factored tensor to free.
*/
void splatt_free_kruskal(
    splatt_kruskal_t * factored);

#ifdef __cplusplus
}
#endif


#endif
