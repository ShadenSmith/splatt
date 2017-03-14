
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../stats.h"
#include "../thd_info.h"
#include "../cpd/cpd.h"
#include "cmd_cpd_constraint.h"






/******************************************************************************
 * HELPER FUNCTIONS
 *****************************************************************************/



/**
* @brief Return an array with the modes specified by the 1-indexed list
*        'args'. If the list is empty, mark all of them.
*
* @param args Array of command line arguments.
* @param num_args On input, the length of 'args'. On output, the length of the
*                 returned list.
*
* @return A list of the specified modes.
*/
static idx_t * p_cmd_parse_modelist(
    char * * args,
    int * num_args)
{
  idx_t * modes = splatt_malloc(MAX_NMODES * sizeof(*modes));

  /* no args? all modes set */
  if(*num_args == 0) {
    *num_args = MAX_NMODES;

    for(idx_t m=0; m < MAX_NMODES; ++m) {
      modes[m] = m;
    }
    return modes;
  }

  for(idx_t m=0; m < MAX_NMODES; ++m) {
    modes[m] = SPLATT_MAX_NMODES;
  }


  /* parse modes */
  int mode_len = 0;
  for(int i=0; i < *num_args; ++i) {
    idx_t mode = strtoull(args[i], NULL, 10) - 1;
    assert(mode < MAX_NMODES);
    
    modes[mode_len++] = mode;
  }

  /* store length */
  *num_args = mode_len;

  return modes;
}


/**
* @brief Split 'cmd_arg' into strings using ',' as a delimiter. split_args[]
*        must already exist and be at least of length MAX_NMODES*2.
*
* @param cmd_arg The string to split.
* @param[out] split_args The list of split strings. NOTE: must be freed.
* @param[out] num_args The length of split_args.
*/
static void p_split_args(
    char * cmd_arg,
    char * * split_args,
    int * num_args)
{
  int N = 0;

  char * ptr = strtok(cmd_arg, ",");
  while(ptr != NULL) {
    /* copy arg */
    split_args[N] = splatt_malloc(strlen(ptr)+1);
    strcpy(split_args[N], ptr);
    ++N;

    /* next arg */
    ptr = strtok(NULL, ",");
  }

  *num_args = N;
}



/**
* @brief Parse an argument from argp and attempt to add a constraint to 'cpd'.
*        Returns true on success.
*
* @param cmd_arg The argument string. E.g., 'l1,1e-2,3,4'.
* @param valid_cmds An array of valid constaints.
* @param[out] cpd The CPD options structure to modify.
*
* @return Whether the addition was successful (if it is well-formed and valid).
*/
static bool p_add_constraint(
    char * cmd_arg,
    constraint_cmd * valid_cmds,
    splatt_cpd_opts * opts)
{
  bool success = true;

  /* split args on ',' */
  char * arg_buf[MAX_NMODES * 2];
  int num_args;
  p_split_args(cmd_arg, arg_buf, &num_args);

  /* +1 and -1 so we skip the name */
  int mode_len = num_args - 1;
  idx_t * modes = p_cmd_parse_modelist(arg_buf+1, &mode_len);

  /* Search for the constraint and call the handle. */
  for(int c=0; valid_cmds[c].name != NULL; ++c) {
    if(strcmp(arg_buf[0], valid_cmds[c].name) == 0) {
      success = valid_cmds[c].handle(opts, modes, mode_len);
      goto CLEANUP;
    }
  }
  /* ERROR */
  fprintf(stderr, "SPLATT: constraint '%s' not found.\n", arg_buf[0]);
  success = false;

  CLEANUP:
  splatt_free(modes);
  for(int c=0; c < num_args; ++c) {
    splatt_free(arg_buf[c]);
  }

  return success;
}



/**
* @brief Parse an argument from argp and attempt to add a regularization to
*        'cpd'.  Returns true on success.
*
* @param cmd_arg The argument string. E.g., 'l1,1e-2,3,4'.
* @param valid_cmds An array of valid regularizations.
* @param[out] cpd The CPD options structure to modify.
*
* @return Whether the addition was successful (if it is well-formed and valid).
*/
static bool p_add_regularization(
    char * cmd_arg,
    regularization_cmd * valid_cmds,
    splatt_cpd_opts * opts)
{
  bool success = true;
  idx_t * modes = NULL;

  /* split args on ',' */
  char * arg_buf[MAX_NMODES * 2];
  int num_args;
  p_split_args(cmd_arg, arg_buf, &num_args);

  if(num_args < 2) {
    fprintf(stderr, "SPLATT: regularizations require 'MULT' parameter.\n");
    success = false;
    goto CLEANUP;
  }

  val_t const multiplier = strtod(arg_buf[1], NULL);

  /* +2 and -2 so we skip the name and multiplier */
  int mode_len = num_args - 2;
  modes = p_cmd_parse_modelist(arg_buf+2, &mode_len);

  /* Search for the regularization and call the handle. */
  for(int c=0; valid_cmds[c].name != NULL; ++c) {
    if(strcmp(arg_buf[0], valid_cmds[c].name) == 0) {
      success = valid_cmds[c].handle(opts, multiplier, modes, mode_len);
      goto CLEANUP;
    }
  }
  /* ERROR */
  fprintf(stderr, "SPLATT: regularization '%s' not found.\n", arg_buf[0]);
  success = false;

  CLEANUP:
  splatt_free(modes);
  for(int c=0; c < num_args; ++c) {
    splatt_free(arg_buf[c]);
  }

  return success;
}



/******************************************************************************
 * SPLATT CPD
 *****************************************************************************/
static char cpd_args_doc[] = "TENSOR";
static char cpd_doc[] =
  "splatt-cpd -- Compute the CPD of a sparse tensor.\n";

/* start above non-printable ASCII */
enum
{
  LONG_SEED = 127,
  LONG_NOWRITE,
  LONG_TILE,
  LONG_TOL,

  LONG_CSF,

  /* constraints and regularizations*/
  LONG_CON,
  LONG_REG,

  /* ADMM tuning */
  LONG_INNER_ITS,
  LONG_INNER_TOL,
};

/* option groupings */
enum
{
  CMD_GROUP_CPD,
  CMD_GROUP_PERFORMANCE,
  CMD_GROUP_CONSTRAINT,
  CMD_GROUP_ADMM,
};


static struct argp_option cpd_options[] = {
  /* override help so we can use -h instead of -? */
  {"help", 'h', 0, 0, "Give this help list."},

  {"seed", LONG_SEED, "SEED", 0, "random seed (default: system time)"},
  {"iters", 'i', "#ITS", 0, "maximum number of outer iterations (default: 200)"},

  {"rank", 'r', "RANK", 0, "rank of factorization (default: 10)"},
  {"stem", 's', "PATH", 0, "file stem for output files (default: ./)"},
  {"nowrite", LONG_NOWRITE, 0, 0, "do not write output to file"},

  {"iters", 'i', "NITERS", 0, "maximum number of iterations to use (default: 50)"},
  {"tol", LONG_TOL, "TOLERANCE", 0, "convergence tolerance (default: 1e-5)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"nowrite", LONG_NOWRITE, 0, 0, "do not write output to file"},
  {"seed", LONG_SEED, "SEED", 0, "random seed (default: system time)"},

  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},

  {"threads", 't', "#THREADS", 0, "number of threads (default: ${OMP_NUM_THREADS})", CMD_GROUP_PERFORMANCE},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"csf", LONG_CSF, "#CSF", 0, "how many CSF to use? {one,two,all} default: two"},
  {"tile", LONG_TILE, 0, 0, "use tiling during MTTKRP", CMD_GROUP_PERFORMANCE},

  {0, 0, 0, 0, CPD_CONSTRAINT_DOC, CMD_GROUP_CONSTRAINT},
  {"con", LONG_CON, "CON[,MODELIST]", 0, "constrained factorization", CMD_GROUP_CONSTRAINT},
  {"reg", LONG_REG, "REG,MULT[,MODELIST]", 0, "regularized factorization", CMD_GROUP_CONSTRAINT},


  {0, 0, 0, 0, "Options for tuning ADMM behavior:", CMD_GROUP_ADMM},
  {"inner-its", LONG_INNER_ITS, "#ITS", 0,
      "maximum number of inner ADMM iterations to use (default: 50)", CMD_GROUP_ADMM},
  {"inner-tol", LONG_INNER_TOL, "TOLERANCE", 0,
      "convergence tolerance of inner ADMM iterations (default: 1e-2)", CMD_GROUP_ADMM},
  {"block", 'b', "#ROWS", 0, "number of rows per ADMM block "
      "(applicable for row-separable constraints; default: 50)", CMD_GROUP_ADMM},
  { 0 }
};


typedef struct
{
  char * ifname;   /** file that we read the tensor from */
  char * stem;   /** file stem */
  bool write;      /** do we write output to file? */
  double * opts;   /** splatt_cpd options */
  idx_t nfactors;

  splatt_global_opts * global_opts;
  splatt_cpd_opts    * cpd_opts;
} cpd_cmd_args;


/**
* @brief Fill a cpd_opts struct with default values.
*
* @param args The cpd_opts struct to fill.
*/
static void default_cpd_opts(
  cpd_cmd_args * args)
{
  args->opts = splatt_default_opts();

  args->global_opts = splatt_alloc_global_opts();
  args->cpd_opts = splatt_alloc_cpd_opts();

  args->ifname    = NULL;
  args->stem = NULL;
  args->write     = true;
  args->nfactors  = 10;
}



static error_t parse_cpd_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  cpd_cmd_args * args = state->input;
  char * buf;
  int cnt = 0;

  /* -i=50 should also work... */
  if(arg != NULL && arg[0] == '=') {
    ++arg;
  }

  val_t scale = 0.;
  idx_t mode = MAX_NMODES;
  idx_t size = 0;

  switch(key) {
  case 'h':
    argp_state_help(state, stdout, ARGP_HELP_STD_HELP);
    break;

  case 'i':
    args->cpd_opts->max_iterations = strtoull(arg, &arg, 10);
    break;
  case LONG_TOL:
    args->cpd_opts->tolerance = atof(arg);
    break;

  case 't':
    args->global_opts->num_threads = atoi(arg);
    splatt_omp_set_num_threads(args->global_opts->num_threads);
    break;

  case 'v':
    timer_inc_verbose();
    if(args->global_opts->verbosity < SPLATT_VERBOSITY_MAX) {
      args->global_opts->verbosity += 1;
    }
    break;

  case LONG_TILE:
    args->opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
    break;
  case LONG_NOWRITE:
    args->write = false;
    break;
  case 's':
    args->stem = arg;
    break;
  case 'r':
    args->nfactors = strtoull(arg, &arg, 10);
    break;

  case LONG_CSF:
    if(strcmp("one", arg) == 0) {
      args->opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
    } else if(strcmp("two", arg) == 0) {
      args->opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_TWOMODE;
    } else if(strcmp("all", arg) == 0) {
      args->opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE;
    } else {
      fprintf(stderr, "SPLATT: --csf option '%s' not recognized.\n", arg);
      argp_usage(state);
    }
    break;

  case LONG_SEED:
    args->global_opts->random_seed = atoi(arg);
    srand(args->global_opts->random_seed);
    break;

  /* ADMM tuning */
  case LONG_INNER_ITS:
    args->cpd_opts->max_inner_iterations = strtoull(arg, &arg, 10);
    break;
  case LONG_INNER_TOL:
    args->cpd_opts->inner_tolerance = atof(arg);
    break;
  case 'b':
    size = strtoull(arg, &arg, 10);
    for(idx_t m=0; m < MAX_NMODES; ++m) {
      args->cpd_opts->chunk_sizes[m] = size;
    }
    break;

  /* constraints */
  case LONG_CON:
    if(!p_add_constraint(arg, constraint_cmds, args->cpd_opts)) {
      argp_state_help(state, stdout, ARGP_HELP_EXIT_ERR);
    }
    break;
  case LONG_REG:
    if(!p_add_regularization(arg, regularization_cmds, args->cpd_opts)) {
      argp_state_help(state, stdout, ARGP_HELP_EXIT_ERR);
    }
    break;


  case ARGP_KEY_ARG:
    if(args->ifname != NULL) {
      argp_usage(state);
      break;
    }
    args->ifname = arg;
    break;
  case ARGP_KEY_END:
    if(args->ifname == NULL) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}

static struct argp cpd_argp =
  {cpd_options, parse_cpd_opt, cpd_args_doc, cpd_doc};




/******************************************************************************
 * SPLATT-CPD
 *****************************************************************************/

int splatt_cpd_cmd2(
  int argc,
  char ** argv)
{
  print_header();

  cpd_cmd_args args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER | ARGP_NO_HELP, 0, &args);

  sptensor_t * tt = tt_read(args.ifname);
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  if(args.global_opts->verbosity > SPLATT_VERBOSITY_NONE) {
    cpd_stats2(args.nfactors, tt->nmodes, args.cpd_opts, args.global_opts);
  }

  splatt_csf * csf = splatt_csf_alloc(tt, args.opts);
  tt_free(tt);

  splatt_kruskal * factored = splatt_alloc_cpd(csf, args.nfactors);

  /* do the factorization */
  splatt_cpd(csf, args.nfactors, args.cpd_opts, args.global_opts, factored);

  /* write output */
  if(args.write) {
    char * lambda_name = splatt_malloc(512 * sizeof(*lambda_name));
    if(args.stem) {
      sprintf(lambda_name, "%s.lambda.mat", args.stem);
    } else {
      sprintf(lambda_name, "lambda.mat");
    }
    vec_write(factored->lambda, args.nfactors, lambda_name);
    splatt_free(lambda_name);


    for(idx_t m=0; m < csf->nmodes; ++m) {
      char * matfname = splatt_malloc(512 * sizeof(*matfname));
      if(args.stem) {
        sprintf(matfname, "%s.mode%"SPLATT_PF_IDX".mat", args.stem, m+1);
      } else {
        sprintf(matfname, "mode%"SPLATT_PF_IDX".mat", m+1);
      }

      matrix_t tmpmat;
      mat_fillptr(&tmpmat, factored->factors[m], csf->dims[m],args.nfactors,1);
      mat_write(&tmpmat, matfname);
      splatt_free(matfname);
    }
  }

  /* cleanup */
  splatt_free_cpd(factored);
  splatt_free_csf(csf, args.opts);
  splatt_free_opts(args.opts);
  splatt_free_cpd_opts(args.cpd_opts);
  splatt_free_global_opts(args.global_opts);

  return EXIT_SUCCESS;
}




