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
  int nvtxs;
  int nedges;
} graph_t;

typedef struct
{
  int nvtxs;
  int nhedges;
  int * vwts;
  int * hewts;
  int * eptr;
  int * eind;
} hgraph_t;


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
hgraph_t * hgraph_fib_alloc(
  ftensor_t const * const ft,
  idx_t const mode);

void hgraph_free(
  hgraph_t * hg);

#endif
