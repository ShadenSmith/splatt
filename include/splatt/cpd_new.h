/**
* @file api_cpd.h
* @brief Functions and structures related to the CPD factorization.
* @author Shaden Smith
* @version 2.0.0
* @date 2016-11-26
*/


#ifndef SPLATT_API_CPD_NEW_H
#define SPLATT_API_CPD_NEW_H



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
  unsigned int row_separable     : 1;

  /** Does the constraint induce sparsity in the factor? May be exploited
   *  during MTTKRP. */
  unsigned int sparsity_inducing : 1;
} splatt_con_hints;


/**
* @brief 
*/
typedef enum
{
  SPLATT_CON_CLOSEDFORM,
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

  /** Initialization function for constraint. This can be used to allocate
   *  buffers and/or manipulate the factor values before the factorization
   *  begins. This function is called after 'vals' has been initialized. */
  void (* init_func) (splatt_val_t * vals,
                      splatt_idx_t const nrows,
                      splatt_idx_t const ncolstype,
                      void ** data);

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
                      int const should_parallelize);


  /** Modify Gram and RHS matrices for a closed-form solve instead of ADMM
   *  iterations. For example, L2 regularization should add to the diagonal
   *  of the gram matrix. Unconstrained factorization does a no-op. */
  void (* clos_func) (splatt_val_t * restrict primal,
                      splatt_val_t * restrict gram,
                      splatt_idx_t const nrows,
                      splatt_idx_t const ncols,
                      void * data);

  /** Cleanup function. This can perform any post-processing on the factor
   *  or deallocate any data used by the constraint. */
  void (* post_func) (splatt_val_t * vals,
                      splatt_idx_t const nrows,
                      splatt_idx_t const ncols,
                      void * data);
} splatt_cpd_constraint;



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif

// splatt_alloc_constraint();
// splatt_register_constraint();
// splatt_free_constraint();


#ifdef __cplusplus
}
#endif

#endif
