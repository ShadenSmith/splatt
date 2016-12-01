/**
* @file cmd_cpd_constraint.h
* @brief Functions for adding constraints/regularizations to the frontend of
*        splatt-cpd.  NOTE: edit this file when adding new constraints.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-12-01
*/


#ifndef SPLATT_CMD_CPD_CONSTRAINT_H
#define SPLATT_CMD_CPD_CONSTRAINT_H




/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../base.h"





/******************************************************************************
 * TYPES
 *****************************************************************************/


/**
* @brief Structure for adding a constraint/regularization during command-line
*        argument parsing.
*/
typedef struct
{
  /** How to specify the constraint/regularization. E.g., --con=<name> */
  char * name;
  /** Function which parses args[:] and adds the constraint to cpd. */
  bool (* handle) (char * * args, int num_args, splatt_cpd_opts * cpd);
} regcon_cmd;







/******************************************************************************
 * BEGIN EDITS
 *****************************************************************************/

/* Just makes function prototypes easier. */
#define PROTO_CONSTRAINT_HANDLE(handle_name) \
    bool handle_name(char * * args, int num_args, splatt_cpd_opts * cpd)


/* FUNCTION PROTOTYPES. ADD YOURS TO THIS LIST. */
//PROTO_CONSTRAINT_HANDLE( ntf_handle );
//PROTO_CONSTRAINT_HANDLE( lasso_handle );



/* CONSTRAINT DOCUMENTATION. ADD TO THIS */
static char const CPD_CONSTRAINT_DOC[] = 
    "\nConstraints and Regularizations\n"
    "-------------------------------\n"
    "splatt uses AO-ADMM [Huang & Sidiropoulos 2015] to enforce constraints "
    "and regularizations. All regularizations require a 'MULT' parameter "
    "which scales the penalty term in the objective function. "
    "All constraints and regularizations optionally accept a 'MODELIST' "
    "parameter as a comma-separated list. When no modes are specified, the "
    "constraint or regularization is applied to all modes.\n\n"

    "NOTE: only one constraint or regularization can be applied per mode. "
    "Existing ones can be overwritten, so '--con=nonneg --reg=smooth,10.0,4' "
    "will find a model with non-negative factors except the fourth, which will "
    "have smooth columns.\n\n"

    "The following constraints are supported:\n"
    "  nonneg\tnon-negativity \n"
    "  ntf\t\t(nonneg alias)\n"
    "  orth\t\torthogonality\n"
    "  rowsimp\trows lie in a probability simplex\n"
    "  colsimp\tcolumns lie in a probability simplex\n"
    "  symm\t\tsymmetry (matching factor matrices, >1 mode required)\n"
    "\n"
    "The following regularizations are supported:\n"
    "  frob\t\tFrobenius norm (Tikhonov regularization)\n"
    "  l1\t\tsparsity (LASSO)\n"
    "  lasso\t(l1 alias)\n"
    "  smooth\tsmooth columns\n";



/* ADD TO THIS LIST. MULTIPLE ENTRIES RESULT IN ALIASES. */
static regcon_cmd constraint_cmds[] = {
  //{"nonneg", ntf_handle} ,
  //{"ntf",    ntf_handle} ,
  //{"l1",    lasso_handle} ,
  //{"lasso", lasso_handle} ,
  { NULL, NULL }
};

/******************************************************************************
 * END EDITS
 *****************************************************************************/







/******************************************************************************
 * HELPER FUNCTIONS
 *****************************************************************************/

/**
* @brief Fill a boolean array with the modes specified by the 1-indexed list
*        'args'. If the list is empty, mark all of them.
*
* @param[out] is_mode_set Marker array for each mode.
* @param args Array of command line arguments.
* @param num_args The length of 'args'.
*/
static void cmd_parse_modelist(
    bool * is_mode_set,
    char * * args,
    int num_args)
{
  /* no args? all modes set */
  if(num_args == 0) {
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      is_mode_set[m] = true;
    }
    return;
  }

  for(idx_t m=0; m < MAX_NMODES; ++m) {
    is_mode_set[m] = false;
  }

  /* parse modes */
  for(int i=0; i < num_args; ++i) {
    idx_t mode = strtoull(args[i], &args[i], 10) - 1;
    assert(mode < MAX_NMODES);

    is_mode_set[mode] = true;
  }
}




#endif
