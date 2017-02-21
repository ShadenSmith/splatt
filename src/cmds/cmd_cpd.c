
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

#define TT_CSF 250
#define TT_REG 251
#define TT_SEED 252
#define TT_NOWRITE 253
#define TT_TOL 254
#define TT_TILE 255
static struct argp_option cpd_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum number of iterations to use (default: 50)"},
  {"tol", TT_TOL, "TOLERANCE", 0, "minimum change for convergence (default: 1e-5)"},
  {"reg", TT_REG, "REGULARIZATION", 0, "regularization parameter (default: 0)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"csf", TT_CSF, "#CSF", 0, "how many CSF to use? {one,two,all} default: two"},
  {"tile", TT_TILE, 0, 0, "use tiling during SPLATT"},
  {"nowrite", TT_NOWRITE, 0, 0, "do not write output to file"},
  {"seed", TT_SEED, "SEED", 0, "random seed (default: system time)"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
  {"stem", 's', "PATH", 0, "file stem for factorization output files (default: ./)"},
  { 0 }
};


typedef struct
{
  char * ifname;   /** file that we read the tensor from */
  char * stem;   /** file stem */
  int write;       /** do we write output to file? */
  double * opts;   /** splatt_cpd options */
  idx_t nfactors;
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
  args->stem = NULL;
  args->ifname    = NULL;
  args->write     = DEFAULT_WRITE;
  args->nfactors  = DEFAULT_NFACTORS;
}


static void free_cpd_args(
  cpd_cmd_args * args)
{
  splatt_free_opts(args->opts);
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
    args->opts[SPLATT_OPTION_NITER] = (double) atoi(arg);
    break;
  case TT_TOL:
    args->opts[SPLATT_OPTION_TOLERANCE] = atof(arg);
    break;
  case TT_REG:
    args->opts[SPLATT_OPTION_REGULARIZE] = atof(arg);
    break;
  case 't':
    args->opts[SPLATT_OPTION_NTHREADS] = (double) atoi(arg);
    splatt_omp_set_num_threads((int)args->opts[SPLATT_OPTION_NTHREADS]);
    break;
  case 'v':
    timer_inc_verbose();
    args->opts[SPLATT_OPTION_VERBOSITY] += 1;
    break;
  case TT_TILE:
    args->opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
    break;
  case TT_NOWRITE:
    args->write = 0;
    break;
  case 'r':
    args->nfactors = atoi(arg);
    break;
  case 's':
    args->stem = arg;
    break;
  case TT_CSF:
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

  case TT_SEED:
    args->opts[SPLATT_OPTION_RANDSEED] = atoi(arg);
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
  srand(args.opts[SPLATT_OPTION_RANDSEED]);

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
    char * lambda_name = NULL;
    if(args.stem) {
      asprintf(&lambda_name, "%s.lambda.mat", args.stem);
    } else {
      asprintf(&lambda_name, "lambda.mat");
    }
    vec_write(factored.lambda, args.nfactors, lambda_name);
    free(lambda_name);

    for(idx_t m=0; m < nmodes; ++m) {
      char * matfname = NULL;
      if(args.stem) {
        asprintf(&matfname, "%s.mode%"SPLATT_PF_IDX".mat", args.stem, m+1);
      } else {
        asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);
      }

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
  free_cpd_args(&args);

  /* free factor matrix allocations */
  splatt_free_kruskal(&factored);

  return EXIT_SUCCESS;
}

