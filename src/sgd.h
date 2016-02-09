#ifndef SPLATT_SGD_H
#define SPLATT_SGD_H

#include "sptensor.h"
#include "kruskal.h"

void splatt_sgd(
    sptensor_t const * const train,
    splatt_kruskal * const model);


#endif
