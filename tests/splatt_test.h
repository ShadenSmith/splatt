#ifndef SPLATT_TEST_H
#define SPLATT_TEST_H

#include "../src/sptensor.h"

/* DATASET(med.tns) will return "/tests/tensors/med.tns" */
#define DATASET_(x) SPLATT_TEST_DATASETS #x
#define DATASET(x) DATASET_(x)


static char const * const datasets[] = {
  DATASET(small.tns),
  DATASET(med.tns),
  DATASET(small4.tns),
  DATASET(med4.tns)
};
#define MAX_DSETS 16

static inline idx_t load_tensors(
  sptensor_t ** tensors)
{
  idx_t ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < ntensors; ++i) {
    tensors[i] = tt_read(datasets[i]);
  }

  return ntensors;
}

static inline void free_tensors(
  sptensor_t ** tensors)
{
  idx_t ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < ntensors; ++i) {
    tt_free(tensors[i]);
  }
}


#endif
