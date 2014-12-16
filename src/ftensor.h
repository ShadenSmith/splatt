#ifndef SPLATT_FTENSOR_H
#define SPLATT_FTENSOR_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"
#include "matrix.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  idx_t nnz;
  idx_t nmodes;
  idx_t dims[MAX_NMODES];

  /* Defines a permutation for each view of ft.
   * Each perm is a list of modes, starting with the mode we are operating on.
   * The first m-1 modes are used to define fibers.
   */
  idx_t dim_perms[MAX_NMODES][MAX_NMODES];

  idx_t  nfibs[MAX_NMODES];
  idx_t * sptr[MAX_NMODES];
  idx_t * fptr[MAX_NMODES];
  idx_t * fids[MAX_NMODES];
  idx_t * inds[MAX_NMODES];
  val_t * vals[MAX_NMODES];
} ftensor_t;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
ftensor_t * ften_alloc(
  sptensor_t * const tt,
  int const tile);

spmatrix_t * ften_spmat(
  ftensor_t * ft,
  idx_t const mode);

void ften_free(
  ftensor_t * ft);

void fib_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const mode,
  idx_t * const perm_dims);

#endif
