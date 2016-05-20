
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../stats.h"
#include "../thd_info.h"
#include "../cpd.h"


/******************************************************************************
 * SPLATT CPD
 *****************************************************************************/
static char cpd_args_doc[] = "TENSOR";
static char cpd_doc[] =
  "splatt-cpd -- Compute the CPD of a sparse tensor.\n";


/* start above non-printable ASCII */
typedef enum
{
  LONG_SEED = 127,
  LONG_NOWRITE,
  LONG_TILE,
  LONG_TOL,

  /* constraints */
  LONG_NONNEG,
} splatt_long_opt;


static struct argp_option cpd_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum number of iterations to use (default: 50)"},
  {"tol", LONG_TOL, "TOLERANCE", 0, "minimum change for convergence (default: 1e-5)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"tile", LONG_TILE, 0, 0, "use tiling during SPLATT"},
  {"nowrite", LONG_NOWRITE, 0, 0, "do not write output to file"},
  {"seed", LONG_SEED, "SEED", 0, "random seed (default: system time)"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},

  {0, 0, 0, 0, "Constraints and Regularizations", 1},

  {"nonneg", LONG_NONNEG, "MODE", OPTION_ARG_OPTIONAL,
      "non-negative factorization (default: all modes).", 1},

  { 0 }
};


typedef struct
{
  char * ifname;   /** file that we read the tensor from */
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

  switch(key) {
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
    args->global_opts->verbosity += 1;
    break;
  case LONG_TILE:
    args->opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
    break;
  case LONG_NOWRITE:
    args->write = false;
    break;
  case 'r':
    args->nfactors = strtoull(arg, &arg, 10);
    break;

  case LONG_SEED:
    args->global_opts->random_seed = atoi(arg);
    srand(args->global_opts->random_seed);
    break;

  /* constraints */
  case LONG_NONNEG:
    if(arg) {
      splatt_cpd_con_nonneg(args->cpd_opts, strtoull(arg, &arg, 10));
    } else {
      splatt_cpd_con_nonneg(args->cpd_opts, MAX_NMODES);
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
int splatt_cpd_cmd(
  int argc,
  char ** argv)
{
  /* assign defaults and parse arguments */
  cpd_cmd_args args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  sptensor_t * tt = NULL;

  print_header();

  tt = tt_read(args.ifname);
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  /* print basic tensor stats? */
  splatt_verbosity_type which_verb = args.opts[SPLATT_OPTION_VERBOSITY];
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);
  }

  splatt_csf * csf = splatt_csf_alloc(tt, args.opts);

  idx_t nmodes = tt->nmodes;
  tt_free(tt);

  /* print CPD stats? */
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    cpd_stats(csf, args.nfactors, args.opts);
  }

  splatt_kruskal factored;

  /* do the factorization! */
  int ret = splatt_cpd_als(csf, args.nfactors, args.opts, &factored);
  if(ret != SPLATT_SUCCESS) {
    fprintf(stderr, "splatt_cpd_als returned %d. Aborting.\n", ret);
    return ret;
  }

  printf("Final fit: %0.5"SPLATT_PF_VAL"\n", factored.fit);

  /* write output */
  if(args.write == 1) {
    vec_write(factored.lambda, args.nfactors, "lambda.mat");

    for(idx_t m=0; m < nmodes; ++m) {
      char * matfname = NULL;
      asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);

      matrix_t tmpmat;
      tmpmat.rowmajor = 1;
      tmpmat.I = csf->dims[m];
      tmpmat.J = args.nfactors;
      tmpmat.vals = factored.factors[m];

      mat_write(&tmpmat, matfname);
      free(matfname);
    }
  }

  /* cleanup */
  splatt_csf_free(csf, args.opts);
  free(args.opts);

  /* free factor matrix allocations */
  splatt_free_kruskal(&factored);

  return EXIT_SUCCESS;
}



int splatt_cpd_cmd2(
  int argc,
  char ** argv)
{
  print_header();

  cpd_cmd_args args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

#if 0
  cpd_opts->constraints[0].which = SPLATT_REG_NONNEG;
  cpd_opts->constraints[1].which = SPLATT_REG_NONNEG;
  cpd_opts->constraints[2].which = SPLATT_REG_NONNEG;
  cpd_opts->constraints[3].which = SPLATT_REG_NONNEG;
  cpd_opts->constraints[4].which = SPLATT_REG_NONNEG;
  cpd_opts->constraints[5].which = SPLATT_REG_NONNEG;
  cpd_opts->constraints[6].which = SPLATT_REG_NONNEG;
  cpd_opts->constraints[1].which = SPLATT_REG_L1;
  val_t * l = splatt_malloc(sizeof(*l));
  *l = 0.10;
  cpd_opts->constraints[1].data = l;
#endif

  sptensor_t * tt = tt_read(args.ifname);
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  double * dopts = splatt_default_opts();

  splatt_csf * csf = splatt_csf_alloc(tt, dopts);
  tt_free(tt);

  splatt_kruskal * factored = splatt_alloc_cpd(csf, args.nfactors);

  splatt_cpd(csf, args.nfactors, args.cpd_opts, args.global_opts, factored);

  /* write output */
  if(args.write) {
    vec_write(factored->lambda, args.nfactors, "lambda.mat");

    for(idx_t m=0; m < csf->nmodes; ++m) {
      char * matfname = NULL;
      asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);

      matrix_t tmpmat;
      tmpmat.rowmajor = 1;
      tmpmat.I = csf->dims[m];
      tmpmat.J = args.nfactors;
      tmpmat.vals = factored->factors[m];

      mat_write(&tmpmat, matfname);
      free(matfname);
    }
  }

  /* cleanup */
  splatt_free_kruskal(factored);
  splatt_free_opts(dopts);
  splatt_free_cpd_opts(args.cpd_opts);
  splatt_free_global_opts(args.global_opts);

  return EXIT_SUCCESS;
}




