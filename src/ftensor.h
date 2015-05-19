#ifndef SPLATT_FTENSOR_H
#define SPLATT_FTENSOR_H

#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  idx_t nnz;
  idx_t nmodes;
  idx_t dims[MAX_NMODES];

  /* Perm is a list of modes, starting with the mode we are operating on.
   * The first m-1 modes are used to define fibers.
   */
  idx_t dim_perms[MAX_NMODES];

  idx_t  nslcs;
  idx_t  nfibs;
  idx_t * sptr;
  idx_t * sids;
  idx_t * fptr;
  idx_t * fids;
  idx_t * inds;
  val_t * vals;

  idx_t * indmap; /** Maps local -> global indices. */

  /* structures for tiled tensors */
  int tiled;
  idx_t    nslabs;
  idx_t * slabptr;
} ftensor_t;


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
ftensor_t * ften_alloc(
  sptensor_t * const tt,
  idx_t const mode,
  int const tile);

#define ften_spmat splatt_ften_spmat
spmatrix_t * ften_spmat(
  ftensor_t * ft);

#define ften_free splatt_ften_free
void ften_free(
  ftensor_t * ft);

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
