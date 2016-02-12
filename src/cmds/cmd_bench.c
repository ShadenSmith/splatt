
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
  "splatt-bench -- benchmark MTTKRP algorithms.\n\n"
  "Available MTTKRP algorithms are:\n"
  "  splatt\tThe algorithm introduced by splatt\n"
  "  csf\t\tGeneralized CSF format\n"
  "  giga\t\tGigaTensor algorithm adapted from the MapReduce paradigm\n"
  "  coord\t\tStream through a coordinate tensor\n"
  "  ttbox\t\tTensor-Vector products as done by Tensor Toolbox\n"
  "Available reordering algorithms are:\n"
  "  graph\t\t\tReorder based on the partitioning of a mode-independent graph\n"
  "  hgraph\t\tReorder based on the partitioning of a hypergraph\n"
  "  fib\t\t'hgraph' reordering AND reschedule fiber execution\n";

typedef enum
{
  ALG_SPLATT,
  ALG_CSF,
  ALG_GIGA,
  ALG_DFACTO,
  ALG_TTBOX,
  ALG_COORD,
  ALG_ERR,
  ALG_NALGS
} splatt_algs;


static void (*bench_funcs[ALG_NALGS]) (sptensor_t * const tt,
                                       matrix_t ** mats,
                                       bench_opts const * const opts)
  = {
    [ALG_SPLATT] = bench_splatt,
    [ALG_CSF]    = bench_csf,
    [ALG_COORD]  = bench_coord,
    [ALG_GIGA]   = bench_giga,
    [ALG_TTBOX]  = bench_ttbox
  };

typedef struct
{
  char * ifname;
  char * pfname;
  splatt_perm_type rtype;
  int which[ALG_NALGS];
  char * algerr;
  char * permerr;
  idx_t niters;
  idx_t nthreads;
  idx_t rank;
  int scale;
  int write;
  int tile;
  idx_t permmode;
} bench_args;

#define TT_TILE 255

static struct argp_option bench_options[] = {
  {"alg", 'a', "ALG", 0, "algorithm to benchmark"},
  {"iters", 'i', "NITERS", 0, "number of iterations to use (default: 5)"},
  {"mode", 'm', "MODE", 0, "mode basis for hgraph reordering (default: 1)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: 1)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"scale", 's', 0, 0, "scale threads from 1 to NTHREADS (by 2)"},
  {"tile", TT_TILE, 0, 0, "use tiling during SPLATT"},
  {"write", 'w', 0, 0, "write results to files ALG_mode<N>.mat (for testing)"},
  {"rtype", 'z', "TYPE", 0, "designate reordering type"},
  {"pfile", 'p', "FILE", 0, "partition file for reordering"},
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
    } else if(strcmp(arg, "csf") == 0) {
      args->which[ALG_CSF] = 1;
    } else if(strcmp(arg, "coord") == 0) {
      args->which[ALG_COORD] = 1;
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
  case 'z':
    if(strcmp(arg, "graph") == 0) {
      args->rtype = PERM_GRAPH;
    } else if(strcmp(arg, "hgraph") == 0) {
      args->rtype = PERM_HGRAPH;
    } else if(strcmp(arg, "fib") == 0) {
      args->rtype = PERM_FIBSCHED;
    } else {
      args->rtype = PERM_ERROR;
      args->permerr = arg;
    }
    break;

  case TT_TILE:
    args->tile = SPLATT_SYNCTILE;
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

static idx_t * p_mkthreads(
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
    tsizes = (idx_t *) splatt_malloc(tcount * sizeof(idx_t));

    for(idx_t t=0; t < tcount; ++t) {
      tsizes[t] = (idx_t) (1 << t);
    }

  } else {
    tcount = 1;
    tsizes = (idx_t *) splatt_malloc(1 * sizeof(idx_t));
    tsizes[0] = nthreads;
  }

  *nruns = tcount;
  return tsizes;
}

int splatt_bench(
  int argc,
  char ** argv)
{
  bench_args args;
  args.ifname = NULL;
  args.pfname = NULL;
  args.algerr = NULL;
  args.permerr = NULL;
  args.niters = 5;
  args.nthreads = 1;
  args.scale = 0;
  args.rank = 10;
  args.write = 0;
  args.tile = 0;
  args.permmode = 0;
  args.rtype = PERM_ERROR;
  for(int a=0; a < ALG_NALGS; ++a) {
    args.which[a] = 0;
  }
  argp_parse(&bench_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  if(args.which[ALG_ERR]) {
    fprintf(stderr, "SPLATT: algorithm '%s' is not recognized.\n"
                    "Run with '--help' for assistance.\n", args.algerr);
    return SPLATT_ERROR_BADINPUT;
  }

  print_header();

  sptensor_t * tt = tt_read(args.ifname);
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  /* fill bench opts */
  bench_opts opts;
  opts.niters = args.niters;
  opts.write = args.write;
  opts.tile = args.tile;

  if(args.pfname != NULL) {
    if(args.rtype == PERM_ERROR) {
      fprintf(stderr, "SPLATT: reordering algorithm '%s' is not recognized.\n"
                      "Run with '--help' for assistance.\n", args.permerr);
      return SPLATT_ERROR_BADINPUT;
    }

    printf("Reordering ------------------------------------------------------\n");
    opts.perm = tt_perm(tt, args.rtype, args.permmode, args.pfname);
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

  opts.threads = p_mkthreads(args.nthreads, args.scale, &opts.nruns);

  printf("Benchmarking ---------------------------------------------------\n");
  printf("RANK=%"SPLATT_PF_IDX" ITS=%"SPLATT_PF_IDX"\n", args.rank, args.niters);

  for(int a=0; a < ALG_NALGS; ++a) {
    if(args.which[a]) {
      bench_funcs[a](tt, mats, &opts);
    }
    printf("\n");
  }

  perm_free(opts.perm);
  free(opts.threads);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mat_free(mats[m]);
  }
  mat_free(mats[MAX_NMODES]);
  tt_free(tt);

  return EXIT_SUCCESS;
}


