
#include "sgd.h"
#include "reorder.h"
#include "util.h"


static val_t const driver_good = 1.10;
static val_t const driver_bad  = 0.50;

void splatt_sgd(
    sptensor_t const * const train,
    splatt_kruskal * const model)
{
  idx_t const nfactors = model->rank;

  val_t const * const restrict train_vals = train->vals;

  idx_t * perm = splatt_malloc(train->nnz * sizeof(*perm));


  val_t lrn_rate = 0.0005;
  val_t reg = 0.1;

  val_t prev_rmse = -1;

  #pragma omp parallel
  {
    val_t * predict_buffer = splatt_malloc(nfactors * sizeof(*predict_buffer));
    val_t * all_others = splatt_malloc(nfactors * sizeof(*all_others));

    /* init perm */
    #pragma omp for schedule(static)
    for(idx_t n=0; n < train->nnz; ++n) {
      perm[n] = n;
    }

    /* foreach epoch */
    for(idx_t e=0; e < 100; ++e) {
      /* new nnz ordering */
      #pragma omp single
      shuffle_idx(perm, train->nnz);
      #pragma omp barrier

      #pragma omp for schedule(static)
      for(idx_t n=0; n < train->nnz; ++n) {
        idx_t const x = perm[n];

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
            update_row[f] += lrn_rate * ((err * all_others[f]) -
                (reg*update_row[f]));
          }
        }

        /* check RMSE at end of epoch */
      }

      /* compute RMSE and adjust learning rate */
      #pragma omp single
      {
        val_t const rmse = kruskal_rmse(train, model);
        if(e % 1 == 0) {
          printf("RMSE: %0.5f\n", rmse);
        }

        if(e > 0) {
          if(rmse < prev_rmse) {
            lrn_rate *= driver_good;
          } else {
            lrn_rate *= driver_bad;
          }
        }
        printf("eta: %0.6f\n", lrn_rate);

        prev_rmse = rmse;
      }

      #pragma omp barrier
    }

    splatt_free(predict_buffer);
    splatt_free(all_others);
  }

  splatt_free(perm);
}


