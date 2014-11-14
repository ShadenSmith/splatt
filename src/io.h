#ifndef SPLATT_IO_H
#define SPLATT_IO_H

#include "sptensor.h"
#include "spmatrix.h"

sptensor_t * tt_read(
  char * const fname);

void tt_write(
  sptensor_t const * const tt,
  char * const fname);

void spmat_write(
  spmatrix_t const * const mat,
  char * const fname);

#endif
