

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../admm.h"





/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


/**
* @brief The proximal update for a non-negative factorization. This routine
*        projects 'primal' onto the non-negative orthant. Simply, it zeroes
*        out negative entries.
*
* @param[out] primal The row-major matrix to update.
* @param nrows The number of rows in primal.
* @param ncols The number of columns in primal.
* @param offset Not used.
* @param data Not used.
* @param rho Not used.
* @param should_parallelize If true, parallelize.
*/
void splatt_nonneg_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize)
{
  #pragma omp parallel for if(should_parallelize)
  for(idx_t i=0; i < nrows; ++i) {
    for(idx_t j=0; j < ncols; ++j) {
      idx_t const index = j + (i*ncols);
      primal[index] = (primal[index] > 0.) ? primal[index] : 0.;
    }
  }
}




/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


splatt_error_type splatt_register_nonneg(
    splatt_cpd_opts * opts,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes)
{
  for(idx_t m = 0; m < num_modes; ++m) {
    idx_t const mode = modes_included[m];

    splatt_cpd_constraint * ntf_con = splatt_alloc_constraint(SPLATT_CON_ADMM);

    /* only fill the details that are used */
    ntf_con->prox_func = splatt_nonneg_prox;

    /* set hints to assist optimizations */
    ntf_con->hints.row_separable     = true;
    ntf_con->hints.sparsity_inducing = true;

    sprintf(ntf_con->description, "NON-NEGATIVE");

    /* add to the CPD factorization */
    splatt_register_constraint(opts, mode, ntf_con);

    /* memory will be freed by splatt_free_constraint() */
  }

  return SPLATT_SUCCESS;
}


