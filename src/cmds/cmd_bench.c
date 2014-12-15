
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../bench.h"
#include "../stats.h"
#include "../reorder.h"



/******************************************************************************
 * SPLATT BENCH
 *****************************************************************************/
static char bench_args_doc[] = "TENSOR [-a ALG]...";
static char bench_doc[] =
  "splatt-bench -- benchmark MTTKRP algorithms\n\n"
  "Available algorithms are:\n"
  "  splatt\tThe algorithm introduced by splatt\n"
  "  giga\t\tGigaTensor algorithm adapted from the MapReduce paradigm\n"
  "  ttbox\t\tTensor-Vector products as done by Tensor Toolbox\n";

typedef enum
{
  ALG_SPLATT,
  ALG_GIGA,
  ALG_DFACTO,
  ALG_TTBOX,
  ALG_ERR,
  ALG_NALGS
} splatt_algs;


static void (*bench_funcs[ALG_NALGS]) (sptensor_t * const tt,
                                       matrix_t ** mats,
                                       bench_opts const * const opts)
  = {
    [ALG_SPLATT] = bench_splatt,
    [ALG_GIGA]   = bench_giga,
    [ALG_TTBOX]  = bench_ttbox
  };

typedef struct
{
  char * ifname;
  char * pfname;
  int which[ALG_NALGS];
  char * algerr;
  idx_t niters;
  idx_t nthreads;
  idx_t rank;
  int scale;
  int write;
  idx_t permmode;
} bench_args;

static struct argp_option bench_options[] = {
  {"alg", 'a', "ALG", 0, "algorithm to benchmark"},
  {"iters", 'i', "NITERS", 0, "number of iterations to use (default: 5)"},
  {"mode", 'm', "MODE", 0, "mode basis for hgraph reordering (default: 1)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: 1)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"scale", 's', 0, 0, "scale threads from 1 to NTHREADS (by 2)"},
  {"write", 'w', 0, 0, "write results to files ALG_mode<N>.mat (for testing)"},
  {"partition", 'p', "FILE", 0, "use an hgraph partitioning to reorder tensor"},
  { 0 }
};


static error_t parse_bench_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  bench_args *args = state->input;
  switch(key) {
  case 'a':
    if(strcmp(arg, "splatt") == 0) {
      args->which[ALG_SPLATT] = 1;
    } else if(strcmp(arg, "giga") == 0) {
      args->which[ALG_GIGA] = 1;
    } else if(strcmp(arg, "dfacto") == 0) {
      args->which[ALG_DFACTO] = 1;
    } else if(strcmp(arg, "ttbox") == 0) {
      args->which[ALG_TTBOX] = 1;
    } else {
      args->which[ALG_ERR] = 1;
      args->algerr = arg;
    }
    break;
  case 'i':
    args->niters = atoi(arg);
    break;
  case 'm':
    args->permmode = atoi(arg) - 1;
    break;
  case 'p':
    args->pfname = arg;
    break;
  case 'r':
    args->rank = atoi(arg);
    break;
  case 's':
    args->scale = 1;
    break;
  case 't':
    args->nthreads = atoi(arg);
    break;
  case 'w':
    args->write = 1;
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

static struct argp bench_argp =
  {bench_options, parse_bench_opt, bench_args_doc, bench_doc};

static idx_t * __mkthreads(
  idx_t const nthreads,
  int const scale,
  idx_t * nruns)
{
  idx_t *tsizes;
  idx_t tcount;

  if(scale) {
    tcount = 1;
    while((idx_t)(1 << tcount) <= nthreads) {
      ++tcount;
    }
    tsizes = (idx_t *) malloc(tcount * sizeof(idx_t));

    for(idx_t t=0; t < tcount; ++t) {
      tsizes[t] = (idx_t) (1 << t);
    }

  } else {
    tcount = 1;
    tsizes = (idx_t *) malloc(1 * sizeof(idx_t));
    tsizes[0] = nthreads;
  }

  *nruns = tcount;
  return tsizes;
}

void splatt_bench(
  int argc,
  char ** argv)
{
  bench_args args;
  args.ifname = NULL;
  args.pfname = NULL;
  args.algerr = NULL;
  args.niters = 5;
  args.nthreads = 1;
  args.scale = 0;
  args.rank = 10;
  args.write = 0;
  args.permmode = 0;
  for(int a=0; a < ALG_NALGS; ++a) {
    args.which[a] = 0;
  }
  argp_parse(&bench_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  if(args.which[ALG_ERR]) {
    fprintf(stderr, "SPLATT: algorithm '%s' is not recognized.\n"
                    "Run with '--help' for assistance.\n", args.algerr);
    exit(EXIT_FAILURE);
  }

  print_header();

  sptensor_t * tt = tt_read(args.ifname);
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  /* fill bench opts */
  bench_opts opts;
  opts.niters = args.niters;
  opts.write = args.write;

  if(args.pfname != NULL) {
    printf("Reordering ------------------------------------------------------\n");
    opts.perm = tt_perm(tt, PERM_HGRAPH, args.permmode, args.pfname);
    printf("\n");
  } else {
    /* initialize perms */
    opts.perm = perm_alloc(tt->dims, 0);
  }

  /* M, the result matrix is stored at mats[MAX_NMODES] */
  idx_t max_dim = 0;
  matrix_t * mats[MAX_NMODES+1];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mats[m] = mat_rand(tt->dims[m], args.rank);
    if(tt->dims[m] > max_dim) {
      max_dim = tt->dims[m];
    }
  }
  mats[MAX_NMODES] = mat_alloc(max_dim, args.rank);

  opts.threads = __mkthreads(args.nthreads, args.scale, &opts.nruns);

  printf("Benchmarking ---------------------------------------------------\n");

  for(int a=0; a < ALG_NALGS; ++a) {
    if(args.which[a]) {
      bench_funcs[a](tt, mats, &opts);
    }
  }

  perm_free(opts.perm);
  free(opts.threads);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mat_free(mats[m]);
  }
  mat_free(mats[MAX_NMODES]);
  tt_free(tt);
}


