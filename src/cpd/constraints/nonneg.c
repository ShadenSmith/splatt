

void prox_nonneg(
    splatt_val_t * primal
    splatt_idx_t const nrows,
    splatt_idx_t const ncols,
    splatt_idx_t const offset,
    void * data,
    splatt_val_t const rho,
    bool const should_parallelize)
{


}



/**
* @brief Project onto the non-negative orthant.
*
* @param[out] mat_primal The primal variable (factor matrix) to update.
* @param penalty The current penalty parameter (rho) -- not used.
* @param ws CPD workspace data -- not used.
* @param mode The mode we are updating -- not used.
* @param data Constraint-specific data -- not used.
* @param is_chunked Whether this is a chunk of a larger matrix.
*/
static void p_proximity_nonneg(
    matrix_t  * const mat_primal,
    matrix_t const * const mat_auxil,
    matrix_t const * const mat_dual,
    val_t const penalty,
    void const * const data,
    bool const should_parallelize)
{
  idx_t const I = mat_primal->I;
  idx_t const J = mat_primal->J;

  val_t       * const restrict matv = mat_primal->vals;
  val_t const * const restrict auxl = mat_auxil->vals;
  val_t const * const restrict dual = mat_dual->vals;

  #pragma omp parallel for schedule(static) if(should_parallelize)
  for(idx_t x=0; x < I * J; ++x) {
    val_t const v = auxl[x] - dual[x];
    matv[x] = (v > 0.) ? v : 0.;
  }
}


void splatt_cpd_con_nonneg(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode)
{
  /* MAX_NMODES will simply apply constraints to all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_con_nonneg(cpd_opts, m);
    }
    return;
  }

  splatt_cpd_con_clear(cpd_opts, mode);

  cpd_opts->unconstrained = false;
  cpd_opts->constraints[mode].which = SPLATT_CON_NONNEG;
}




bool ntf_handle(
    char * * args,
    int num_args,
    char * * state)
{
  bool modes[MAX_NMODES];
  con_parse_modes(modes, args, num_args);

  /* now apply NTF */
  for(int m=0; m < MAX_NMODES; ++m) {
    if(modes[m]) {
      state[m] = malloc(4);
      sprintf(state[m], "NTF");
    }
  }

  return true;
}

