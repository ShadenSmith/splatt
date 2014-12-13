#ifndef SPLATT_REORDER_H
#define SPLATT_REORDER_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"

/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef enum
{
  PERM_GRAPH,
  PERM_HGRAPH,
  PERM_ERROR
} splatt_perm_type;

typedef struct
{
  idx_t * perms[MAX_NMODES];
  idx_t * iperms[MAX_NMODES];
} perm_t;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_perm(
  sptensor_t * const tt,
  splatt_perm_type const type,
  idx_t const mode,
  char const * const pfile);

void build_pptr(
  idx_t const * const parts,
  idx_t const nparts,
  idx_t const nvtxs,
  idx_t ** ret_pptr,
  idx_t ** ret_plookup);


/******************************************************************************
 * MATRIX REORDER FUNCTIONS
 *****************************************************************************/
matrix_t * perm_matrix(
  matrix_t const * const mat,
  idx_t const * const perm,
  matrix_t * retmat);



#endif
