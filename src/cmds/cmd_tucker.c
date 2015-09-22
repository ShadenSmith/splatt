
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../stats.h"

/******************************************************************************
 * ARG PARSING
 *****************************************************************************/
static char tucker_args_doc[] = "TENSOR";
static char tucker_doc[] =
  "splatt-tucker -- Compute the Tucker Decomposition of a sparse tensor.\n";


#define TT_NOWRITE 253
#define TT_TOL 254
static struct argp_option tucker_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum number of iterations to use (default: 50)"},
  {"tol", TT_TOL, "TOLERANCE", 0, "minimum change for convergence (default: 1e-5)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"nowrite", TT_NOWRITE, 0, 0, "do not write output to file (default: WRITE)"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
  { 0 }
};

typedef struct
{
  char * ifname;   /** file that we read the tensor from */
  int write;       /** do we write output to file? */
  double * opts;   /** splatt options */
  idx_t nfactors;
} tucker_cmd_args;


/**
* @brief Fill a tucker_cmd_args struct with default values.
*
* @param args The tucker_opts struct to fill.
*/
static void default_tucker_opts(
  tucker_cmd_args * args)
{
  args->opts = splatt_default_opts();
  args->ifname    = NULL;
  args->write     = DEFAULT_WRITE;
  args->nfactors  = DEFAULT_NFACTORS;
}


static error_t parse_tucker_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  tucker_cmd_args * args = state->input;
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
    timer_inc_verbose();
    args->opts[SPLATT_OPTION_VERBOSITY] += 1;
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

static struct argp tucker_argp =
  {tucker_options, parse_tucker_opt, tucker_args_doc, tucker_doc};

/******************************************************************************
 * SPLATT TUCKER
 *****************************************************************************/
void splatt_tucker_cmd(
  int argc,
  char ** argv)
{
  /* assign defaults and parse arguments */
  tucker_cmd_args args;
  default_tucker_opts(&args);
  argp_parse(&tucker_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  print_header();

  sptensor_t * tt = NULL;
  tt = tt_read(args.ifname);
  if(tt == NULL) {
    return;
  }
  tt_remove_empty(tt);
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  tt_free(tt);
}

