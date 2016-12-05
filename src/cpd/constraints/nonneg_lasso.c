

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../admm.h"





/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

/**
* @brief The proximal update for a L1-regularized factorization (LASSO) via
*        soft thresholding with lambda/rho.
*
* @param[out] primal The row-major matrix to update.
* @param nrows The number of rows in primal.
* @param ncols The number of columns in primal.
* @param offset Not used.
* @param data Not used.
* @param rho Not used.
* @param should_parallelize If true, parallelize.
*/
void ntf_lasso_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize)
{
  val_t const lambda = *((val_t *) data);
  val_t const mult = lambda / rho;

  #pragma omp parallel for schedule(static) if(should_parallelize)
  for(idx_t x=0; x < nrows * ncols; ++x) {
    val_t const v = primal[x];
    primal[x] = (v > mult) ? (v - mult) : 0.;
  }
}


/**
* @brief Free the single val_t allocated for L1 regularization.
*
* @param data The data to free.
*/
void ntf_lasso_free(
    void * data)
{
  splatt_free(data);
}




/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

splatt_error_type splatt_register_ntf_lasso(
    splatt_cpd_opts * opts,
    splatt_val_t const multiplier,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes)
{
  splatt_cpd_constraint * lasso_con = NULL;
  for(idx_t m = 0; m < num_modes; ++m) {
    idx_t const mode = modes_included[m];

    lasso_con = splatt_alloc_constraint(SPLATT_CON_ADMM);

    lasso_con->prox_func = ntf_lasso_prox;
    lasso_con->free_func = ntf_lasso_free;

    /* important hints */
    lasso_con->hints.row_separable     = true;
    lasso_con->hints.sparsity_inducing = true;

    sprintf(lasso_con->description, "NTF-L1-REG (%0.1e)", multiplier);

    /* store multiplier */
    val_t * mult = splatt_malloc(sizeof(*mult));
    *mult = multiplier;
    lasso_con->data = mult;

    /* store the constraint for use */
    splatt_register_constraint(opts, mode, lasso_con);
  }
    
  return SPLATT_SUCCESS;
}

