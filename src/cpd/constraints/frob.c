



/**
* @brief Optimally compute the new primal variable when no regularization or
*        only L2 regularization is present.
*
* @param[out] primal The matrix to update.
* @param ws CPD workspace.
* @param which_reg Which regularization we are using.
* @param cdata The constraint data -- 'lambda' if L2 regularization is used.
*/
static void p_admm_optimal_regs(
    matrix_t * const primal,
    cpd_ws * const ws,
    splatt_con_type which_reg,
    void const * const cdata)
{
  /* Add to the diagonal for L2 regularization. */
  if(which_reg == SPLATT_REG_L2) {
    val_t const reg = *((val_t *) cdata);
    mat_add_diag(ws->gram, reg);
  }

  mat_cholesky(ws->gram);

  /* Copy and then solve directly against MTTKRP */
  size_t const bytes = primal->I * primal->J * sizeof(*primal->vals);
  par_memcpy(primal->vals, ws->mttkrp_buf->vals, bytes);
  mat_solve_cholesky(ws->gram, primal);
}


void splatt_cpd_reg_l2(
    splatt_cpd_opts * const cpd_opts,
    splatt_idx_t const mode,
    splatt_val_t const scale)
{
  /* MAX_NMODES will simply apply regularization to all modes */
  if(mode == MAX_NMODES) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      splatt_cpd_reg_l2(cpd_opts, m, scale);
    }
    return;
  }

  //splatt_cpd_con_clear(cpd_opts, mode);

  cpd_opts->constraints[mode].which = SPLATT_REG_L2;
  val_t * lambda = splatt_malloc(sizeof(*lambda));
  *lambda = scale;
  cpd_opts->constraints[mode].data = (void *) lambda;
}



