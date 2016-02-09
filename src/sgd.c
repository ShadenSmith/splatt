
#include "sgd.h"

static val_t const eta = 0.0007;
static val_t const reg = 0.3;

void splatt_sgd(
    sptensor_t const * const train,
    splatt_kruskal * const model)
{
  idx_t const nfactors = model->rank;

  val_t const * const restrict train_vals = train->vals;

  #pragma omp parallel
  {
    val_t * predict_buffer = splatt_malloc(nfactors * sizeof(*predict_buffer));
    val_t * all_others = splatt_malloc(nfactors * sizeof(*all_others));

    /* foreach epoch */
    for(idx_t e=0; e < 100; ++e) {
      #pragma omp for schedule(static)
      for(idx_t x=0; x < train->nnz; ++x) {
        val_t const err = train_vals[x] - predict_val(model, train, x,
            predict_buffer);
        //printf("%0.2f vs %0.2f\n", train_vals[x], predict_val(model, train, x,
            //predict_buffer));

        for(idx_t m=0; m < train->nmodes; ++m) {
          val_t * const restrict update_row = model->factors[m] +
              (train->ind[m][x] * nfactors);

          for(idx_t f=0; f < nfactors; ++f) {
            all_others[f] = 1.;
          }

          for(idx_t m2=0; m2 < train->nmodes; ++m2) {
            if(m2 != m) {
              val_t const * const restrict row = model->factors[m2] +
                  (train->ind[m2][x] * nfactors);
              for(idx_t f=0; f < nfactors; ++f) {
                all_others[f] *= row[f];
              }
            }
          }

          for(idx_t f=0; f < nfactors; ++f) {
            update_row[f] += eta * ((err * all_others[f]) - (reg*update_row[f]));
          }
        }

        /* check RMSE at end of epoch */
      }

      #pragma omp barrier
      #pragma omp single
      if(e % 5 == 0) {
        printf("RMSE: %0.5f\n", kruskal_rmse(train, model));
      }
      #pragma omp barrier
    }

    splatt_free(predict_buffer);
    splatt_free(all_others);
  }
}


