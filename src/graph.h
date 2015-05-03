#ifndef SPLATT_GRAPH_H
#define SPLATT_GRAPH_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "ftensor.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  idx_t nvtxs;
  idx_t nedges;
} graph_t;

typedef struct
{
  idx_t nvtxs;
  idx_t nhedges;
  idx_t * vwts;
  idx_t * hewts;
  idx_t * eptr;
  idx_t * eind;
} hgraph_t;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
#define hgraph_fib_alloc splatt_hgraph_fib_alloc
hgraph_t * hgraph_fib_alloc(
  ftensor_t const * const ft,
  idx_t const mode);


#define hgraph_free splatt_hgraph_free
void hgraph_free(
  hgraph_t * hg);


#define hgraph_uncut splatt_hgraph_uncut
idx_t * hgraph_uncut(
  hgraph_t const * const hg,
  idx_t const * const parts,
  idx_t * const ncut);

#endif
