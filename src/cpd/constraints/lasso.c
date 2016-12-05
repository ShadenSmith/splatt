
/**
* @brief Perform Lasso regularization via soft thresholding.
*
* @param[out] mat_primal The primal variable (factor matrix) to update.
* @param penalty The current penalty parameter (rho).
* @param ws CPD workspace data -- used for regularization parameter.
* @param mode The mode we are updating.
* @param data Constraint-specific data -- we store lambda in this.
* @param is_chunked Whether this is a chunk of a larger matrix.
*/
static void p_proximity_l1(
    matrix_t  * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t const * const mat_dual,
    val_t const penalty,
    void const * const data,
    bool const is_chunked)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxl = mat_auxil->vals;
  val_t const * const restrict dual = mat_dual->vals;

  val_t const lambda = *((val_t *) data);
  val_t const mult = lambda / penalty;

  #pragma omp parallel for schedule(static) if(!is_chunked)
  for(idx_t x=0; x < I * J; ++x) {
    val_t const v = auxl[x] - dual[x];

    /* TODO: could this be done faster? */
    if(v > mult) {
      matv[x] = v - mult;
    } else if(v < -mult) {
      matv[x] = v + mult;
    } else {
      matv[x] = 0.;
    }
  }
}


void splatt_cpd_reg_l1(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale)
{
  /* MAX_NMODES will simply apply regularization to all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_reg_l1(cpd_opts, m, scale);
    }
    return;
  }

  splatt_cpd_con_clear(cpd_opts, mode);

  cpd_opts->unconstrained = false;
  cpd_opts->constraints[mode].which = SPLATT_REG_L1;
  val_t * lambda = splatt_malloc(sizeof(*lambda));
  *lambda = scale;
  cpd_opts->constraints[mode].data = (void *) lambda;
}


