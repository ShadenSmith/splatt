#ifndef SPLATT_FTENSOR_H
#define SPLATT_FTENSOR_H

#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
#define ftensor_t splatt_csf_t


/* 1 selects longer fiber direction, 0 chooses short fibers */
#ifndef SPLATT_LONG_FIB
#define SPLATT_LONG_FIB 1
#endif



/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "sptensor.h"
#include "matrix.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
#define ften_alloc splatt_ften_alloc
void ften_alloc(
  ftensor_t * const ft,
  sptensor_t * const tt,
  idx_t const mode,
  int const tile);


#define ften_spmat splatt_ften_spmat
spmatrix_t * ften_spmat(
  ftensor_t * ft);


#define ften_free splatt_ften_free
void ften_free(
  ftensor_t * const ft);


#define fib_mode_order splatt_fib_mode_order
void fib_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const mode,
  idx_t * const perm_dims);


#define ften_storage splatt_ften_storage
/**
* @brief Return the number of bytes that ft uses.
*
* @param ft The FTensor to analyze.
*
* @return The number of bytes of storage.
*/
size_t ften_storage(
  ftensor_t const * const ft);

#endif
