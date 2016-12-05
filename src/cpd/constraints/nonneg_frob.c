

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../admm.h"





/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


/**
* @brief Interpret data as a scalar to add to the Gram diagonal. The scalar is
*        \lambda, the penalty multiplier on the Tikhonov regularization.
*
* @param[out] gram The Gram matrix to update.
* @param N The dimension of the matrix.
* @param data Interpreted as a val_t scalar.
*/
void nonneg_frob_gram(
    val_t * restrict gram,
    splatt_idx_t const N,
    void * data)
{
  /* convert (void *) to val_t */
  val_t const mult = *((val_t *) data);

  /* add mult to diagonal */
  for(idx_t n=0; n < N; ++n) {
    gram[n + (n * N)] += mult;
  }
}



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
void nonneg_frob_prox(
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

    ntf_con->gram_func = nonneg_frob_gram;
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


