

#include "../../base.h"


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
void splatt_prox_nonneg(
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



/**
* @brief Register a non-negativity constraint with a list of modes.
*
* @param[out] opts The CPD options structure to modify.
* @param modes_included A list of length at least SPLATT_MAX_NMODES.
*     Non-negativity is imposed on mode 'm' if modes_included[m] == true.
*
* @return True. There are no error conditions to handle.
*/
bool splatt_register_nonneg(
    splatt_cpd_opts * opts,
    bool const * const modes_included)
{
  for(idx_t mode = 0; mode < SPLATT_MAX_NMODES; ++mode) {
    if(!modes_included[mode]) {
      continue;
    }

    splatt_cpd_constraint * ntf_con = splatt_alloc_constraint(SPLATT_CON_ADMM);

    /* only fill the details that are used */
    ntf_con->prox_func = splatt_prox_nonneg;

    /* set hints to assist optimizations */
    ntf_con->hints.row_separable     = true;
    ntf_con->hints.sparsity_inducing = true;

    /* add to the CPD factorization */
    splatt_register_constraint(opts, mode, ntf_con);

    /* memory will be freed by splatt_free_constraint() */
  }

  return true;
}


