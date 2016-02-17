#ifndef SPLATT_SGD_H
#define SPLATT_SGD_H

#include "sptensor.h"
#include "kruskal.h"

void splatt_sgd(
    sptensor_t const * const train,
    sptensor_t const * const validate,
    splatt_kruskal * const model,
    idx_t const max_epochs,
    val_t learn_rate,
    val_t const * const regularization);

#endif
