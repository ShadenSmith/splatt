/**
* @file api_cpd.h
* @brief Functions and structures related to the CPD factorization.
* @author Shaden Smith
* @version 2.0.0
* @date 2016-05-14
*/


#ifndef SPLATT_API_CPD_H
#define SPLATT_API_CPD_H


/******************************************************************************
 * TYPES
 *****************************************************************************/


/**
* @brief Provide hints about a constraint which may be utilized to improve
*        performance.
*/
typedef struct
{
  /** Is the proximity operator row-separable? Aids in parallelization. */
  bool row_separable;

  /** Does the constraint induce sparsity in the factor? May be exploited
   *  during MTTKRP. */
  bool sparsity_inducing;
} splatt_con_hints;


/**
* @brief How will the constraint/regularization be enforced?
*/
typedef enum
{
  /** Constraint/regularization has a closed form solution. */
  SPLATT_CON_CLOSEDFORM,
  /** Use ADMM. */
  SPLATT_CON_ADMM
} splatt_con_solve_type;


/**
* @brief A constraint or regularization imposed during a CPD factorization.
*/
typedef struct
{
  /** Whether there is a closed form solution or ADMM iterations are used. */
  splatt_con_solve_type solve_type;

  /** Performance hints. */
  splatt_con_hints hints;

  /** Arbitrary data. Often a penalty weight or other. */
  void * data;

  /** String description of the constraint. E.g., 'NON-NEGATIVE' or 'LASSO'. */
  char description[256];

  /** Initialization function for constraint. This can be used to allocate
   *  buffers and/or manipulate the factor values before the factorization
   *  begins. This function is called after 'vals' has been initialized. */
  void (* init_func) (splatt_val_t * vals,
                      splatt_idx_t const nrows,
                      splatt_idx_t const ncols,
                      void * data);

  /**
  * @brief Modify the Gram matrix before computing a Cholesky factorization.
  *
  * @param[out] gram The NxN Gram matrix to modify.
  * @param N The dimension of the Gram matrix.
  * @param data Private constraint data.
  */
  void (* gram_func) (splatt_val_t * gram,
                      splatt_idx_t const N,
                      void * data);

  /** Apply the proximity operator to the (nrows x ncols) primal matrix. This
   *  may be a submatrix which starts at row 'offset'. 'rho' is the multiplier
   *  for the current ADMM iteration which some constraints need to consider.
   *  For example, L1 regularization should use soft thresholding around
   *  lambda/rho. */
  void (* prox_func) (splatt_val_t * primal,
                      splatt_idx_t const nrows,
                      splatt_idx_t const ncols,
                      splatt_idx_t const offset,
                      void * data,
                      splatt_val_t const rho,
                      bool const should_parallelize);



  /**
  * @brief Setup a the primal (RHS) matrix before forward/backward solves
  *        during a closed form solve.
  *
  * @param[out] primal The matrix to setup.
  * @param nrows The number of rows.
  * @param ncols The number of columns.
  * @param data Private constraint data.
  */
  void (* clsd_func) (splatt_val_t * primal,
                      splatt_idx_t const nrows,
                      splatt_idx_t const ncols,
                      void * data);

  /** Post-processing function. This will be called after the factorization has
   *  completed. */
  void (* post_func) (splatt_val_t * vals,
                      splatt_idx_t const nrows,
                      splatt_idx_t const ncols,
                      void * data);
  
  /** Deallocation function. This should deallocate any data. */
  void (* free_func) (void * data);

} splatt_cpd_constraint;




typedef struct
{
  /* convergence */
  splatt_val_t tolerance;
  splatt_idx_t max_iterations;

  /* inner ADMM solves */
  splatt_val_t inner_tolerance;
  splatt_idx_t max_inner_iterations;

  /* constraints */
  splatt_cpd_constraint * constraints[SPLATT_MAX_NMODES];

  /* chunked AO-ADMM. 0=unchunked */
  splatt_idx_t chunk_sizes[SPLATT_MAX_NMODES];
} splatt_cpd_opts;




/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif


/**
 * \defgroup constraint_api Functions for configuring a CPD factorization.
 *  @{
 */


/**
* @brief Register a non-negativity constraint with a list of modes.
*
* @param[out] opts The CPD options structure to modify.
* @param modes_included A list of the modes to register.
* @param num_modes The length of 'modes_included'.
*
* @return SPLATT error code.
*/
splatt_error_type splatt_register_nonneg(
    splatt_cpd_opts * opts,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes);


/**
* @brief Register a row simplex constraint with a list of modes. Rows of the
*        affected factor will be projected onto the probability simplex. This
*        means that they will have non-negative values, and rows will sum to
*        one.
*
* @param[out] opts The CPD options structure to modify.
* @param modes_included A list of the modes to register.
* @param num_modes The length of 'modes_included'.
*
* @return SPLATT error code.
*/
splatt_error_type splatt_register_rowsimp(
    splatt_cpd_opts * opts,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes);


/**
* @brief Register a column simplex constraint with a list of modes. Columns of
*        the affected factor will be projected onto the probability simplex.
*        This means that they will have non-negative values, and columns will
*        sum to one.
*
* @param[out] opts The CPD options structure to modify.
* @param modes_included A list of the modes to register.
* @param num_modes The length of 'modes_included'.
*
* @return SPLATT error code.
*/
splatt_error_type splatt_register_colsimp(
    splatt_cpd_opts * opts,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes);


/**
* @brief Register a Frobenius norm (Tiknohov) regularization with a list of
*        modes.
*
* @param[out] opts The CPD options structure to modify.
* @param multiplier The \lambda penalty multiplier.
* @param modes_included A list of the modes to register.
* @param num_modes The length of 'modes_included'.
*
* @return SPLATT error code.
*/
splatt_error_type splatt_register_frob(
    splatt_cpd_opts * opts,
    splatt_val_t const multiplier,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes);


/**
* @brief Register an L1-norm (LASSO) regularization with a list of
*        modes.
*
* @param[out] opts The CPD options structure to modify.
* @param multiplier The \lambda penalty multiplier.
* @param modes_included A list of the modes to register.
* @param num_modes The length of 'modes_included'.
*
* @return SPLATT error code.
*/
splatt_error_type splatt_register_lasso(
    splatt_cpd_opts * opts,
    splatt_val_t const multiplier,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes);


/**
* @brief Register an column smoothness regularization with a list of modes.
*
* @param[out] opts The CPD options structure to modify.
* @param multiplier The \lambda penalty multiplier.
* @param modes_included A list of the modes to register.
* @param num_modes The length of 'modes_included'.
*
* @return SPLATT error code.
*/
splatt_error_type splatt_register_smooth(
    splatt_cpd_opts * opts,
    splatt_val_t const multiplier,
    splatt_idx_t const * const modes_included,
    splatt_idx_t const num_modes);

/** }@ */





/**
 * \defgroup CPD_opts Functions for configuring a CPD factorization.
 *  @{
 */

/**
* @brief Allocate a `splatt_cpd_opts` structure and initialize with sane
*        defaults.
*
* @return The allocated options, to be freed by `splatt_free_cpd_opts()`.
*/
splatt_cpd_opts * splatt_alloc_cpd_opts(void);

/**
* @brief Free the memory allocated by `splatt_alloc_cpd_opts()`.
*
* @param opts The structure to free.
*/
void splatt_free_cpd_opts(
    splatt_cpd_opts * const opts);



/**
* @brief Allocate and initialize a constraint structure. All fields are
*        initalized to 'identity' values, meaning no additional work is
*        performed during factorization unless specified.
*
* @param solve_type How the constraint will be enforced.
*
* @return A constraint structure, to be freed by splatt_free_constraint().
*/
splatt_cpd_constraint * splatt_alloc_constraint(
    splatt_con_solve_type const solve_type);



/**
* @brief Deallocate any memory associated with a constraint. If con->free_func
*        is non-NULL, it will also be called.
*
* @param con The constraint to free.
*/
void splatt_free_constraint(
    splatt_cpd_constraint * const con);


/**
* @brief Register a constraint with a mode of a CPD.
*
* @param opts Specifications for the CPD.
* @param mode The mode to associate the constraint with.
* @param con A constraint allocated by splatt_alloc_constraint().
*/
void splatt_register_constraint(
    splatt_cpd_opts * const opts,
    splatt_idx_t const mode,
    splatt_cpd_constraint * const con);


/** }@ */



/**
* @brief Allocate the memory required for a CPD factorization of a given tensor.
*
* @param tensor The tensor to factor.
* @param rank The desired factorization rank.
*
* @return Allocated factors.
*/
splatt_kruskal * splatt_alloc_cpd(
    splatt_csf const * const tensor,
    splatt_idx_t rank);


/**
* @brief Free a splatt_kruskal allocated by `splatt_alloc_cpd()`.
*
* @param factored The factored tensor to free.
*/
void splatt_free_cpd(
    splatt_kruskal * factored);


/**
* @brief Compute a CPD factorization.
*
* @param tensor The tensor to factor.
* @param rank The desired factorization rank.
* @param cpd_opts Configuration for the specific factorization. Can be NULL for
*                 default options.
* @param global_opts Configuration for general SPLATT behavior. Can be NULL for
*                    default options.
* @param[out] factored The factored tensor.
*
* @return SPLATT error code. SPLATT_SUCCESS if no error.
*
* @ingroup splatt_factorizations
*/
splatt_error_type splatt_cpd(
    splatt_csf const * const tensor,
    splatt_idx_t rank,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts,
    splatt_kruskal * factored);


#ifdef __cplusplus
}
#endif


#endif
