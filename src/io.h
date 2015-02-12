#ifndef SPLATT_IO_H
#define SPLATT_IO_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "matrix.h"
#include "graph.h"


/**
* @brief Open a file.
*
* @param fname The name of the file.
* @param mode The mode for opening.
*
* @return A FILE pointer.
*/
static inline FILE * open_f(
  char const * const fname,
  char const * const mode)
{
  FILE * f;
  if((f = fopen(fname, mode)) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n", fname);
    exit(1);
  }
  return f;
}

/******************************************************************************
 * TENSOR FUNCTIONS
 *****************************************************************************/
sptensor_t * tt_read_file(
  char const * const fname);

void tt_write_file(
  sptensor_t const * const tt,
  FILE * fout);

void tt_write(
  sptensor_t const * const tt,
  char const * const fname);


/******************************************************************************
 * GRAPH FUNCTIONS
 *****************************************************************************/
void hgraph_write_file(
  hgraph_t const * const hg,
  FILE * fout);
void hgraph_write(
  hgraph_t const * const hg,
  char const * const fname);



/******************************************************************************
 * DENSE MATRIX FUNCTIONS
 *****************************************************************************/
void mat_write(
  matrix_t const * const mat,
  char const * const fname);



/******************************************************************************
 * SPARSE MATRIX FUNCTIONS
 *****************************************************************************/
void spmat_write(
  spmatrix_t const * const mat,
  char const * const fname);
void spmat_write_file(
  spmatrix_t const * const mat,
  FILE * fout);



/******************************************************************************
 * PARTITION FUNCTIONS
 *****************************************************************************/
idx_t * part_read(
  char const * const ifname,
  idx_t const nvtxs,
  idx_t * nparts);

#endif
