

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../admm.h"





/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


/**
* @brief The proximal update for a non-negative factorization. This routine
*        projects 'primal' onto the non-negative orthant while adding a Frob.
*        norm regularizer. This scales primal by inv((rho/(lambda+rho)) * eye),
*        or more simply divides each entry by (rho/(lambda+rho)). It then
*        projects to the non-negative orthant.
*
* @param[out] primal The row-major matrix to update.
* @param nrows The number of rows in primal.
* @param ncols The number of columns in primal.
* @param offset Not used.
* @param data Not used.
* @param rho Not used.
* @param should_parallelize If true, parallelize.
*/
void nonneg_frob_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize)
{
  val_t const lambda = *((val_t *) data);
  val_t const mult = (lambda + rho) / rho;

  #pragma omp parallel for if(should_parallelize)
  for(idx_t i=0; i < nrows; ++i) {
    for(idx_t j=0; j < ncols; ++j) {
      idx_t const index = j + (i*ncols);
      val_t const new_val = primal[index] * mult;
      primal[index] = (new_val > 0.) ? new_val : 0.;
    }
  }
}


/**
* @brief Free the single val_t allocated for Frobenius regularization.
*
* @param data The data to free.
*/
void nonneg_frob_free(
    void * data)
{
  splatt_free(data);
}



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


splatt_error_type splatt_register_ntf_frob(
    splatt_cpd_opts * opts,
    splatt_val_t const multiplier,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes)
{
  for(idx_t m = 0; m < num_modes; ++m) {
    idx_t const mode = modes_included[m];

    splatt_cpd_constraint * ntf_con = splatt_alloc_constraint(SPLATT_CON_ADMM);

    ntf_con->prox_func = nonneg_frob_prox;
    ntf_con->free_func = nonneg_frob_free;

    /* set hints to assist optimizations */
    ntf_con->hints.row_separable     = true;
    ntf_con->hints.sparsity_inducing = true;

    sprintf(ntf_con->description, "NTF-L1-REG (%0.1e)", multiplier);

    /* store multiplier */
    val_t * mult = splatt_malloc(sizeof(*mult));
    *mult = multiplier;
    ntf_con->data = mult;

    /* add to the CPD factorization */
    splatt_register_constraint(opts, mode, ntf_con);

    /* memory will be freed by splatt_free_constraint() */
  }

  return SPLATT_SUCCESS;
}


