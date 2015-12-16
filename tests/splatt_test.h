#ifndef SPLATT_TEST_H
#define SPLATT_TEST_H

/* DATASET(med.tns) will return "/tests/tensors/med.tns" */
#define DATASET_(x) SPLATT_TEST_DATASETS #x
#define DATASET(x) DATASET_(x)

#define GRAPH_(x) SPLATT_TEST_GRAPHS #x
#define GRAPH(x) GRAPH_(x)

static char const * const datasets[] = {
  DATASET(small.tns),
  DATASET(med_a.tns),
  DATASET(med_b.tns),
  DATASET(med_c.tns),
  DATASET(small4.tns),
  DATASET(med4.tns)
};
#define MAX_DSETS 16


static char const * const graphs[] = {
  GRAPH(small.graph),
  GRAPH(med_a.graph),
  GRAPH(med_b.graph),
  GRAPH(med_c.graph),
  GRAPH(small4.graph),
  GRAPH(med4.graph)
};
#define MAX_GRAPHS 16

#endif
