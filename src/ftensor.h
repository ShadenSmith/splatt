#ifndef SPLATT_FTENSOR_H
#define SPLATT_FTENSOR_H



/*
 * DEPRECATED
 */





#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

/* 1 selects longer fiber direction, 0 chooses short fibers */
#ifndef SPLATT_LONG_FIB
#define SPLATT_LONG_FIB 1
#endif


/**
* @brief Struct describing SPLATT's compressed sparse fiber (CSF) format. Use
*        the splatt_csf_* functions to allocate, fill, and free this structure.
*/
typedef struct ftensor_t
{
  splatt_idx_t nnz;                     /** Number of nonzeros */
  splatt_idx_t nmodes;                  /** Number of modes */
  splatt_idx_t dims[SPLATT_MAX_NMODES];        /** Dimension of each mode */
  splatt_idx_t dim_perm[SPLATT_MAX_NMODES];    /** Permutation of modes */

  splatt_idx_t  nslcs;    /** Number of slices (length of sptr) */
  splatt_idx_t  nfibs;    /** Number of fibers (length of fptr) */
  splatt_idx_t * sptr;    /** Indexes into fptr the start/end of each slice */
  splatt_idx_t * fptr;    /** Indexes into vals the start/end of each fiber */
  splatt_idx_t * fids;    /** ID of each fiber (for mode dim_perm[nmodes-2])*/
  splatt_idx_t * inds;    /** ID of each nnz (for dim_perm[nmodes-1]) */
  splatt_val_t * vals;    /** Floating point value of each nonzero */

  splatt_idx_t * indmap;  /** Maps local to global indices if empty slices */

  /* TILED STRUCTURES */
  splatt_tile_type tiled;
  splatt_idx_t    nslabs;   /** Number of slabs (length of slabptr) */
  splatt_idx_t * slabptr;   /** Indexes into fptr the start/end of each slab */
  splatt_idx_t * sids;      /** ID of each fiber (for dim_perm[0]) */
} ftensor_t;




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
