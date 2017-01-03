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
  /** Function which adds a constraint to modes in modes_included[:]. */
  splatt_error_type (* handle) (
                   splatt_cpd_opts * opts,
                   idx_t const * const modes_included,
                   idx_t const num_modes);
} constraint_cmd;


/**
* @brief Structure for adding a regularization during command-line argument
*        parsing.
*/
typedef struct
{
  /** How to specify the regularization. E.g., --reg=<name> */
  char * name;
  /** Function which adds a constraint to modes in modes_included[:]. */
  splatt_error_type (* handle) (
                   splatt_cpd_opts * opts,
                   val_t const multiplier,
                   idx_t const * const modes_included,
                   idx_t const num_modes);
} regularization_cmd;






/******************************************************************************
 * BEGIN EDITS
 *****************************************************************************/

/* Just makes function prototypes easier. */
#define PROTO_CONSTRAINT_HANDLE(handle_name) \
    splatt_error_type handle_name( \
                   splatt_cpd_opts * cpd, \
                   idx_t const * const modes_included, \
                   idx_t const num_modes)

#define PROTO_REGULARIZATION_HANDLE(handle_name) \
    splatt_error_type handle_name( \
                   splatt_cpd_opts * cpd, \
                   val_t const multiplier, \
                   idx_t const * const modes_included, \
                   idx_t const num_modes)


/* FUNCTION PROTOTYPES. ADD YOURS HERE OR 'include/splatt/cpd.h'. */
PROTO_REGULARIZATION_HANDLE( splatt_register_ntf_frob );
PROTO_REGULARIZATION_HANDLE( splatt_register_ntf_lasso );



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
    "  rowsimp\trows lie in a probability simplex\n"
    "  colsimp\tcolumns lie in a probability simplex\n"
#if 0
    "  orth\t\torthogonality\n"
    "  symm\t\tsymmetry (matching factor matrices, >1 mode required)\n"
#endif
    "\n"
    "The following regularizations are supported:\n"
    "  frob\t\tFrobenius norm (Tikhonov regularization)\n"
    "  l1\t\tsparsity (LASSO)\n"
    "  lasso\t(l1 alias)\n"
    "  ntf-frob\tFrobenius norm with non-negativity constraint\n"
    "  ntf-l1\tL1 with non-negativity constraint\n"
    "  smooth\tsmooth columns\n"
    "";



/* ADD TO THESE LISTS. MULTIPLE ENTRIES RESULT IN ALIASES. */
static constraint_cmd constraint_cmds[] = {
  {"nonneg", splatt_register_nonneg},
  {"ntf",    splatt_register_nonneg},
  {"rowsimp",    splatt_register_rowsimp},
  {"colsimp",    splatt_register_colsimp},
  { NULL, NULL }
};


static regularization_cmd regularization_cmds[] = {
  {"frob",  splatt_register_frob},
  {"l1",    splatt_register_lasso},
  {"lasso", splatt_register_lasso},
  {"ntf-frob", splatt_register_ntf_frob},
  {"ntf-l1", splatt_register_ntf_lasso},
  {"smooth", splatt_register_smooth},
  { NULL, NULL }
};

/******************************************************************************
 * END EDITS
 *****************************************************************************/

#endif
