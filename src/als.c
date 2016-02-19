
#include "sgd.h"


void splatt_als(
    sptensor_t const * const train,
    sptensor_t const * const validate,
    splatt_kruskal * const model,
    idx_t const max_epochs,
    val_t learn_rate,
    val_t const * const regularization)
{
  idx_t const nfactors = model->rank;

  printf("ALS\n");

  /* outer iterations */
  for(idx_t e=0; e < max_epochs; ++e) {
    for(idx_t m=0; m < train->nmodes; ++m) {

    }
  }
}


