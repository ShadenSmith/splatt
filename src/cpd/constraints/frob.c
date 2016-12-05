
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
void frob_gram(
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
* @brief Free the single (val_t *) allocated for Frobenius regularization.
*
* @param data The data to free.
*/
void frob_free(
    void * data)
{
  splatt_free(data);
}





/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


splatt_error_type splatt_register_frob(
    splatt_cpd_opts * opts,
    splatt_val_t const multiplier,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes)
{
  splatt_cpd_constraint * frob_con = NULL;
  for(idx_t m = 0; m < num_modes; ++m) {
    idx_t const mode = modes_included[m];

    frob_con = splatt_alloc_constraint(SPLATT_CON_CLOSEDFORM);

    /* Tikhonov regularization only requires a modified Gram matrix. */
    frob_con->gram_func = frob_gram;

    /* Remember to clean up */
    frob_con->free_func = frob_free;

    /* Not actually used because it has a closed form solution. But, this is
     * still a good hint to provide for future proofing. */
    frob_con->hints.row_separable = true;

    sprintf(frob_con->description, "FROBENIUS-REG (%0.1e)", multiplier);

    /* store multiplier */
    val_t * mult = splatt_malloc(sizeof(*mult));
    *mult = multiplier;
    frob_con->data = mult;

    /* store the constraint for use */
    splatt_register_constraint(opts, mode, frob_con);
  }

  return SPLATT_SUCCESS;
}




