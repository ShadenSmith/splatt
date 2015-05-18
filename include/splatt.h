#ifndef SPLATT_SPLATT_H
#define SPLATT_SPLATT_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <stddef.h>


/******************************************************************************
 * TYPES
 *****************************************************************************/
typedef double splatt_val_t;
#define SS_VAL "f"
#define SS_MPI_VAL MPI_DOUBLE

typedef unsigned long splatt_idx_t;
#define SS_IDX "lu"
#define SS_MPI_IDX MPI_UNSIGNED_LONG


/******************************************************************************
 * VERSION
 *****************************************************************************/
#define SPLATT_VER_MAJOR     0
#define SPLATT_VER_MINOR     0
#define SPLATT_VER_SUBMINOR  1



/******************************************************************************
 * CONSTANTS
 *****************************************************************************/

typedef enum
{
  SPLATT_SUCCESS = 1,
  SPLATT_ERROR_BADINPUT,
  SPLATT_ERROR_NOMEMORY
} splatt_error_t;


typedef enum
{
  SPLATT_VERBOSITY_NONE,
  SPLATT_VERBOSITY_LOW,
  SPLATT_VERBOSITY_HIGH,
  SPLATT_VERBOSITY_MAX
} splatt_verbosity_t;


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

#ifdef __cplusplus
extern 'C' {
#endif

/**
* @brief Compute the CPD using alternating least squares.
*
* @param rank The rank of the decomposition to perform.
* @param nmodes The number of modes in the tensor. Optimizations are currently
*               only present for nmodes=3.
* @param nnz The number of nonzeros in the tensor.
* @param inds The nonzero indices of the tensor. The nth nonzero can be found
*             at inds[0][n], inds[1][n], ... , inds[m][n]. These indices
*             WILL be rearranged during computation (for sorting, etc.).
* @param vals The nonzero values of the tensor. These values WILL be rearranged
*             during computation (for sorting. etc.).
* @param mats The factor matrices, pre-initialized. Layout is assumed to be
*             row-major.
* @param lambda The scaling factors extracted from mats.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_cpd(
    splatt_idx_t const rank,
    splatt_idx_t const nmodes,
    splatt_idx_t const nnz,
    splatt_idx_t ** const inds,
    splatt_val_t * const vals,
    splatt_val_t ** const mats,
    splatt_val_t * const lambda);

#ifdef __cplusplus
}
#endif


#endif
