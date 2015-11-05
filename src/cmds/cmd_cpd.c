
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../stats.h"
#include "../cpd.h"


/******************************************************************************
 * SPLATT CPD
 *****************************************************************************/
static char cpd_args_doc[] = "TENSOR";
static char cpd_doc[] =
  "splatt-cpd -- Compute the CPD of a sparse tensor.\n";

#define TT_NOWRITE 253
#define TT_TOL 254
#define TT_TILE 255
static struct argp_option cpd_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum number of iterations to use (default: 50)"},
  {"tol", TT_TOL, "TOLERANCE", 0, "minimum change for convergence (default: 1e-5)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"tile", TT_TILE, 0, 0, "use tiling during SPLATT"},
  {"nowrite", TT_NOWRITE, 0, 0, "do not write output to file"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
  { 0 }
};


typedef struct
{
  char * ifname;   /** file that we read the tensor from */
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
  args->ifname    = NULL;
  args->write     = DEFAULT_WRITE;
  args->nfactors  = DEFAULT_NFACTORS;
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
  case 't':
    args->opts[SPLATT_OPTION_NTHREADS] = (double) atoi(arg);
    break;
  case 'v':
    timer_lvl = TIMER_LVL2;
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
void splatt_cpd_cmd(
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
    return;
  }

  /* print basic tensor stats? */
  splatt_verbosity_t which_verb = args.opts[SPLATT_OPTION_VERBOSITY];
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);
  }

  splatt_csf * csf = csf_alloc(tt, args.opts);

  idx_t nmodes = tt->nmodes;
  tt_free(tt);

  /* print CPD stats? */
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    cpd_stats(csf, args.nfactors, args.opts);
  }

  splatt_kruskal_t factored;

  /* do the factorization! */
  int ret = splatt_cpd_als(csf, args.nfactors, args.opts, &factored);
  if(ret != SPLATT_SUCCESS) {
    fprintf(stderr, "splatt_cpd_als returned %d. Aborting.\n", ret);
    exit(1);
  }

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
  csf_free(csf, args.opts);
  free(args.opts);

  /* free factor matrix allocations */
  splatt_free_kruskal(&factored);

}

