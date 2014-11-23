#ifndef SPLATT_IO_H
#define SPLATT_IO_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "matrix.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
sptensor_t * tt_read_file(
  FILE * fin);
sptensor_t * tt_read(
  char const * const fname);

void mat_write(
  matrix_t const * const mat,
  char const * const fname);

void tt_write_file(
  sptensor_t const * const tt,
  FILE * fout);
void tt_write(
  sptensor_t const * const tt,
  char const * const fname);

void spmat_write(
  spmatrix_t const * const mat,
  char const * const fname);

#endif
