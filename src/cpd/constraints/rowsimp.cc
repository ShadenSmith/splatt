

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

extern "C" {
#include "../admm.h"
}

#include <math.h>
#include <algorithm>
#include <functional>



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


extern "C"
/**
* @brief The proximal update for a row simplex constraint. Since rows are
*        assumed to be small, the naive (ncols * log(ncols)) algorithm is used.
*
* @param[out] primal The row-major matrix to update.
* @param nrows The number of rows in primal.
* @param ncols The number of columns in primal.
* @param offset Not used.
* @param data Not used.
* @param rho Not used.
* @param should_parallelize If true, parallelize.
*/
void splatt_rowsimp_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize)
{
  #pragma omp parallel if(should_parallelize)
  {
    val_t * restrict row_buf = (val_t *) splatt_malloc(ncols * sizeof(val_t));

    #pragma omp for
    for(idx_t i=0; i < nrows; ++i) {
      val_t * const restrict row = primal + (i * ncols);

      /* sort row into non-increasing order */
      for(idx_t j=0; j < ncols; ++j) {
        row_buf[j] = row[j];
      }
      std::sort(row_buf, row_buf + ncols, std::greater<val_t>());

      val_t running_sum = -1.; /* only -1 once */
      idx_t pivot;
      for(pivot = 0; pivot < ncols; ++pivot) {
        running_sum += row_buf[pivot];

        val_t const v = row_buf[pivot] - (running_sum / ((val_t) (pivot+1)));

        /* pivot is j-1, so adjust accordingly */
        if(v <= 0.) {
          running_sum -= row_buf[pivot];
          break;
        }
      }
      val_t const theta = running_sum / ((val_t) pivot);

      /* update row */
      for(idx_t j=0; j < ncols; ++j) {
        row[j] -= theta;
        row[j] = SS_MAX(row[j], 0.);
      }
    }

    splatt_free(row_buf);
  } /* end omp parallel */
}



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


splatt_error_type splatt_register_rowsimp(
    splatt_cpd_opts * opts,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes)
{
  for(idx_t m = 0; m < num_modes; ++m) {
    idx_t const mode = modes_included[m];

    splatt_cpd_constraint * con = splatt_alloc_constraint(SPLATT_CON_ADMM);

    /* only fill the details that are used */
    con->prox_func = splatt_rowsimp_prox;

    /* set hints to assist optimizations */
    con->hints.row_separable     = true;
    con->hints.sparsity_inducing = true;

    sprintf(con->description, "ROW-SIMPLEX");

    /* add to the CPD factorization */
    splatt_register_constraint(opts, mode, con);
  }

  return SPLATT_SUCCESS;
}


