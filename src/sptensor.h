#ifndef SPLATT_SPTENSOR_H
#define SPLATT_SPTENSOR_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"



/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef enum
{
  SPLATT_NMODE,
  SPLATT_3MODE,
} tt_type;

typedef struct
{
  tt_type type;
  idx_t nnz;
  idx_t nmodes;
  idx_t * dims;
  idx_t ** ind;
  val_t * vals;
} sptensor_t;



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

sptensor_t * tt_read(
  char const * const ifname);

sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes);

void tt_free(
  sptensor_t * tt);

spmatrix_t * tt_unfold(
  sptensor_t * const tt,
  idx_t const mode);

#endif
