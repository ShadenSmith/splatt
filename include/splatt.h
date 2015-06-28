#ifndef SPLATT_SPLATT_H
#define SPLATT_SPLATT_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <stddef.h>
#include <float.h>


/******************************************************************************
 * TYPES
 *****************************************************************************/

typedef unsigned long splatt_idx_t;
typedef double        splatt_val_t;


/******************************************************************************
 * VERSION
 *****************************************************************************/
#define SPLATT_VER_MAJOR     0
#define SPLATT_VER_MINOR     0
#define SPLATT_VER_SUBMINOR  1



/******************************************************************************
 * ENUMS / CONSTANTS
 *****************************************************************************/


/**
* @brief Return codes used by SPLATT.
*/
typedef enum
{
  SPLATT_SUCCESS = 1,     /** Successful SPLATT API call. */
  SPLATT_ERROR_BADINPUT,  /** SPLATT found an issue with the input.
                              Try splatt-check to debug. */
  SPLATT_ERROR_NOMEMORY   /** SPLATT did not have enough memory to complete.
                              Try using fewer factors, less precision, or a
                              smaller tensor. */
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
  SPLATT_COOPTILE
} splatt_tile_t;


/**
* @brief Enum for defining SPLATT options.
*/
typedef enum
{
  SPLATT_OPTION_TOLERANCE,  /** Threshold for convergence. */
  SPLATT_OPTION_NITER,      /** Maximum number of iterations to perform. */
  SPLATT_OPTION_TILE,       /** Use cache tiling during MTTKRP. */
  SPLATT_OPTION_NTHREADS,   /** Number of OpenMP threads to use. */
  SPLATT_OPTION_RANDSEED,   /** Random number seed */
  SPLATT_OPTION_VERBOSITY,  /** Verbosity level */
  SPLATT_OPTION_NOPTIONS    /** Gives the size of the options array. */
} splatt_option_t;

static double const SPLATT_VAL_OFF = -DBL_MAX;



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

#ifdef __cplusplus
extern 'C' {
#endif

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


/**
* @brief Compute the CPD using alternating least squares.
*
* @param nfactors The rank of the decomposition to perform.
* @param nmodes The number of modes in the tensor. Optimizations are currently
*               only present for nmodes=3.
* @param nnz The number of nonzeros in the tensor.
* @param inds The nonzero indices of the tensor. The nth nonzero can be found
*             at inds[0][n], inds[1][n], ... , inds[m][n]. These indices
*             WILL be rearranged during computation (for sorting, etc.).
* @param vals The nonzero values of the tensor. These values WILL be rearranged
*             during computation (for sorting. etc.).
* @param mats The factor matrices, pre-allocated. Layout is assumed to be
*             row-major.
* @param lambda The scaling factors extracted from mats.
* @param options Options array for SPLATT.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_cpd(
    splatt_idx_t const nfactors,
    splatt_idx_t const nmodes,
    splatt_idx_t const nnz,
    splatt_idx_t ** const inds,
    splatt_val_t * const vals,
    splatt_val_t ** const mats,
    splatt_val_t * const lambda,
    double const * const options);

#ifdef __cplusplus
}
#endif


#endif
