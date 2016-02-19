#ifndef SPLATT_SGD_H
#define SPLATT_SGD_H

#include "sptensor.h"
#include "completion.h"

void splatt_sgd(
    sptensor_t const * const train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws);


#if 0
void splatt_als(
    sptensor_t const * const train,
    sptensor_t const * const validate,
    tc_model * const model,
    idx_t const max_epochs,
    val_t learn_rate,
    val_t const * const regularization);
#endif

#endif
