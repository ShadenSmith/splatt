#ifndef SPLATT_MATRIX_H
#define SPLATT_MATRIX_H

#include "base.h"

typedef struct
{
  idx_t I;
  idx_t J;
  val_t *vals;
} matrix_t;

#endif
