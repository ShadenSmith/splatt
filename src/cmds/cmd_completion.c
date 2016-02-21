
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../completion/completion.h"
#include "../stats.h"
#include <omp.h>



/******************************************************************************
 * ARG PARSING
 *****************************************************************************/

static char tc_args_doc[] = "<train> <validate> [test]";
static char tc_doc[] =
  "splatt-complete -- Complete a tensor with missing entries.\n"
  "Available tensor completion algorithms are:\n"
  " sgd\t\tgradient descent\n"
  "  sgd\t\tstochastic gradient descent\n"
  "  als\t\talternating least squares\n";


#define TC_REG 255
#define TC_NOWRITE 254
static struct argp_option tc_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum iterations/epochs (default: 500)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
  {"alg", 'a', "ALG", 0, "which opt algorithm to use (default: sgd)"},
  {"nowrite", TC_NOWRITE, 0, 0, "do not write output to file"},
  {"step", 's', "SIZE", 0, "step size (learning rate) for SGD"},
  {"reg", TC_REG, "SIZE", 0, "step size (learning rate) for SGD"},
  {0}
};


typedef struct
{
  char * name;
  splatt_tc_type which;
} tc_alg_map;

static tc_alg_map maps[] = {
  { "gd", SPLATT_TC_GD },
  { "sgd", SPLATT_TC_SGD },
  { "als", SPLATT_TC_ALS },
  { NULL,  SPLATT_TC_NALGS }
};


/**
* @brief Parse a command into a splatt_tc_type (completion algorithm).
*
* @param arg The string to parse.
*
* @return The optimization algorithm to use. SPLATT_TC_NALGS on error.
*/
static splatt_tc_type parse_tc_alg(
    char const * const arg)
{
  int ptr = 0;
  while(maps[ptr].name != NULL) {
    if(strcmp(arg, maps[ptr].name) == 0) {
      return maps[ptr].which;
    }
    ++ptr;
  }

  /* error */
  return SPLATT_TC_NALGS;
}



typedef struct
{
  /* TRAIN, VALIDATE, TEST */
  char * ifnames[3];
  int nfiles;

  splatt_tc_type which_alg;

  bool write;

  val_t learn_rate;
  val_t reg;

  idx_t max_its;
  idx_t nfactors;
  idx_t nthreads;
} tc_cmd_args;


/**
* @brief Fill the tensor completion arguments with some sane defaults.
*
* @param args The arguments to initialize.
*/
static void default_tc_opts(
    tc_cmd_args * const args)
{
  args->nfiles = 0;
  for(int n=0; n < 3; ++n) {
    args->ifnames[n] = NULL;
  }

  args->which_alg = SPLATT_TC_SGD;
  args->write = false;
  args->nfactors = 10;
  args->learn_rate = -1.;
  args->reg = -1.;
  args->max_its = 0;
  args->nthreads = omp_get_num_procs();
}



static error_t parse_tc_opt(
    int key,
    char * arg,
    struct argp_state * state)
{
  tc_cmd_args * args = state->input;
  char * buf;
  int cnt = 0;

  /* -i=50 should also work... */
  if(arg != NULL && arg[0] == '=') {
    ++arg;
  }

  switch(key) {
  case 'i':
    args->max_its = strtoull(arg, &buf, 10);
    break;
  case 't':
    args->nthreads = strtoull(arg, &buf, 10);
    omp_set_num_threads(args->nthreads);
    break;
  case 'v':
    timer_inc_verbose();
    break;
  case 'a':
    args->which_alg = parse_tc_alg(arg);
    if(args->which_alg == SPLATT_TC_NALGS) {
      fprintf(stderr, "SPLATT: unknown completion algorithm '%s'.\n", arg);
      argp_usage(state);
    }
    break;
  case TC_NOWRITE:
    args->write = false;
    break;
  case 'r':
    args->nfactors = strtoull(arg, &buf, 10);
    break;
  case 's':
    args->learn_rate = strtod(arg, &buf);
    break;
  case TC_REG:
    args->reg = strtod(arg, &buf);
    break;

  case ARGP_KEY_ARG:
    if(args->nfiles == 3) {
      argp_usage(state);
      break;
    }
    args->ifnames[args->nfiles++] = arg;
    break;

  case ARGP_KEY_END:
    if(args->ifnames[0] == NULL || args->ifnames[1] == NULL) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}


/* tie it all together */
static struct argp tc_argp = {tc_options, parse_tc_opt, tc_args_doc, tc_doc};


/******************************************************************************
 * SPLATT-COMPLETE
 *****************************************************************************/
int splatt_tc_cmd(
  int argc,
  char ** argv)
{
  tc_cmd_args args;
  default_tc_opts(&args);
  argp_parse(&tc_argp, argc, argv, ARGP_IN_ORDER, 0, &args);
  print_header();

  sptensor_t * train = tt_read(args.ifnames[0]);
  sptensor_t * validate = tt_read(args.ifnames[1]);
  if(train == NULL || validate == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }
  idx_t const nmodes = train->nmodes;

  /* print basic tensor stats */
  stats_tt(train, args.ifnames[0], STATS_BASIC, 0, NULL);

  printf("validate nnz: %"SPLATT_PF_IDX"\n\n", validate->nnz);

  /* allocate model + workspace */
  tc_model * model = tc_model_alloc(train, args.nfactors, args.which_alg);
  tc_ws * ws = tc_ws_alloc(model, args.nthreads);

  /* check for non-default vals */
  if(args.learn_rate != -1.) {
    ws->learn_rate = args.learn_rate;
  }
  if(args.reg != -1.) {
    for(idx_t m=0; m < nmodes; ++m) {
      ws->regularization[m] = args.reg;
    }
  }
  if(args.max_its != 0) {
    ws->max_its = args.max_its;
  }

  printf("lrn: %0.3e  reg: %0.3e\n\n", ws->learn_rate, ws->regularization[0]);

  switch(args.which_alg) {
  case SPLATT_TC_GD:
    splatt_tc_gd(train, validate, model, ws);
    break;
  case SPLATT_TC_SGD:
    splatt_tc_sgd(train, validate, model, ws);
    break;
  case SPLATT_TC_ALS:
    splatt_tc_als(train, validate, model, ws);
    break;
  default:
    /* error */
    fprintf(stderr, "SPLATT: unknown completion algorithm\n");
    return SPLATT_ERROR_BADINPUT;
  }

  tt_free(validate);
  tt_free(train);

  /* test rmse */
  if(args.ifnames[2] != NULL) {
    sptensor_t * test = tt_read(argv[3]);
    if(test == NULL) {
      return SPLATT_ERROR_BADINPUT;
    }
    printf("test nnz: %"SPLATT_PF_IDX"\n", test->nnz);
    printf("TEST RMSE: %0.5f\n", tc_rmse(test, model, ws));
    tt_free(test);
  }

  /* write output */
  if(args.write) {
    for(idx_t m=0; m < nmodes; ++m) {
      char * matfname = NULL;
      asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);

      matrix_t tmpmat;
      tmpmat.rowmajor = 1;
      tmpmat.I = model->dims[m];
      tmpmat.J = args.nfactors;
      tmpmat.vals = model->factors[m];

      mat_write(&tmpmat, matfname);
      free(matfname);
    }
  }

  tc_model_free(model);
  tc_ws_free(ws);

  return EXIT_SUCCESS;
}


