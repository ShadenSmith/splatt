#ifndef SPLATT_SPLATT_TYPES_H
#define SPLATT_SPLATT_TYPES_H

/******************************************************************************
 * TYPES
 *****************************************************************************/

/* USER: You may edit values to chance the size of integer and real types.
 * Accepted values are 32 and 64. Changing these values to 32 will decrease
 * memory consumption at the cost of precision and maximum supported tensor
 * size. */

#ifndef SPLATT_IDX_TYPEWIDTH
  #define SPLATT_IDX_TYPEWIDTH 64
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
  #error "*** Incorrect user-supplied value of SPLATT_IDX_TYPEWIDTH ***"
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
  #error "*** Incorrect user-supplied value of SPLATT_VAL_TYPEWIDTH ***"
#endif





/******************************************************************************
 * ENUMS
 *****************************************************************************/


/**
* @brief Enum for defining SPLATT options. Use the splatt_default_opts() and
*        splatt_free_opts() functions to initialize and free an options array.
*/
typedef enum
{
  /* high level options */
  SPLATT_OPTION_NTHREADS,   /* Number of OpenMP threads to use. */
  SPLATT_OPTION_TOLERANCE,  /* Threshold for convergence. */
  SPLATT_OPTION_REGULARIZE, /* Regularization parameter. */
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


#endif
