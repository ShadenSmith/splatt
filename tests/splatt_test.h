#ifndef SPLATT_TEST_H
#define SPLATT_TEST_H

#include "../src/sptensor.h"

/* DATASET(med.tns) will return "/tests/tensors/med.tns" */
#define DATASET_(x) SPLATT_TEST_DATASETS #x
#define DATASET(x) DATASET_(x)


static char const * const datasets[] = {
  DATASET(small.tns),
  DATASET(med.tns),
  DATASET(small4.tns)
};

#endif
