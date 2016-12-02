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
* @brief Structure for adding a constraint during command-line argument
*        parsing.
*/
typedef struct
{
  /** How to specify the constraint. E.g., --con=<name> */
  char * name;
  /** Function which adds a constraint to modes marked in modes_included[:]. */
  bool (* handle) (splatt_cpd_opts * opts, bool * modes_included);
} constraint_cmd;


/**
* @brief Structure for adding a regularization during command-line argument
*        parsing.
*/
typedef struct
{
  /** How to specify the regularization. E.g., --reg=<name> */
  char * name;
  /** Function which adds a regularization to modes marked in
   *  modes_included[:]. */
  bool (* handle) (splatt_cpd_opts * opts, bool * modes_included, val_t mult);
} regularization_cmd;






/******************************************************************************
 * BEGIN EDITS
 *****************************************************************************/

/* Just makes function prototypes easier. */
#define PROTO_CONSTRAINT_HANDLE(handle_name) \
    bool handle_name(splatt_cpd_opts * cpd, bool * modes_included)

#define PROTO_REGULARIZATION_HANDLE(handle_name) \
    bool handle_name(splatt_cpd_opts * cpd, bool * modes_included, val_t mult)


/* FUNCTION PROTOTYPES. ADD YOURS TO THIS LIST. */
PROTO_CONSTRAINT_HANDLE( splatt_register_nonneg );

//PROTO_REGULARIZATION_HANDLE( lasso_handle );



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



/* ADD TO THESE LISTS. MULTIPLE ENTRIES RESULT IN ALIASES. */
static constraint_cmd constraint_cmds[] = {
  {"nonneg", splatt_register_nonneg} ,
  {"ntf",    splatt_register_nonneg} ,
  { NULL, NULL }
};


static regularization_cmd regularization_cmds[] = {
  //{"l1",    lasso_handle} ,
  //{"lasso", lasso_handle} ,
  { NULL, NULL }
};

/******************************************************************************
 * END EDITS
 *****************************************************************************/

#endif
