
#include "base.h"

#include <argp.h>

/* SPLATT MODULES */
#include "timer.h"
#include "io.h"
#include "convert.h"
#include "stats.h"
#include "bench.h"

/* DATA STRUCTURES */
#include "sptensor.h"
#include "matrix.h"


/******************************************************************************
 * SPLATT GLOBAL INFO
 *****************************************************************************/
char const *argp_program_version = "splatt v0.0";
char const *argp_program_bug_address = "Shaden Smith <shaden@cs.umn.edu>";


/******************************************************************************
 * SPLATT CMDS
 *****************************************************************************/
typedef enum splatt_cmd
{
  CMD_CPD,
  CMD_BENCH,
  CMD_CONVERT,
  CMD_PERM,
  CMD_STATS,
  CMD_HELP,
  CMD_ERROR
} splatt_cmd;

typedef struct splatt_args
{
  char * cmd_str;
  splatt_cmd cmd;
} splatt_args;

static char cmd_args_doc[] = "CMD";
static char cmd_doc[] =
  "splatt -- the Surprisingly ParalleL spArse Tensor Toolkit\n\n"
  "The available commands are:\n"
  "  cpd\t\tCompute the Canonical Polyadic Decomposition\n"
  "  bench\t\tBenchmark MTTKRP algorithms\n"
  "  convert\tConvert a tensor to different formats\n"
  "  perm\t\tPermute a tensor using one of several methods\n"
  "  stats\t\tPrint tensor statistics\n"
  "  help\t\tPrint this help message\n";

void cmd_not_implemented(char const * const cmd)
{
  printf("SPLATT: option '%s' is not yet implemented.\n", cmd);
  exit(1);
}

static splatt_cmd read_cmd(char const * const str)
{
  splatt_cmd cmd = CMD_ERROR;
  if(strcmp(str, "cpd") == 0) {
    cmd = CMD_CPD;
  } else if(strcmp(str, "bench") == 0) {
    cmd = CMD_BENCH;
  } else if(strcmp(str, "convert") == 0) {
    cmd = CMD_CONVERT;
  } else if(strcmp(str, "perm") == 0) {
    cmd = CMD_PERM;
  } else if(strcmp(str, "stats") == 0) {
    cmd = CMD_STATS;
  } else if(strcmp(str, "help") == 0) {
    cmd = CMD_HELP;
  }
  return cmd;
}

static error_t parse_cmd(
  int key,
  char * arg,
  struct argp_state * state)
{
  splatt_args *args = state->input;
  switch(key) {
  case ARGP_KEY_ARG:
    if(state->arg_num == 0) {
      args->cmd_str = arg;
      args->cmd = read_cmd(arg);
      if(args->cmd == CMD_ERROR || args->cmd == CMD_HELP) {
        argp_state_help(state, stdout, ARGP_HELP_STD_HELP);
      }
    }
    break;
  case ARGP_KEY_END:
    if(state->arg_num < 1) {
      argp_state_help(state, stdout, ARGP_HELP_STD_HELP);
      break;
    }
  }
  return 0;
}
static struct argp cmd_argp = { 0, parse_cmd, cmd_args_doc, cmd_doc };


/******************************************************************************
 * SPLATT STATS
 *****************************************************************************/
static char stats_args_doc[] = "TENSOR";
static char stats_doc[] =
  "splatt-stats -- print statistics about a tensor\n\n"
  "Mode-independent types are:\n"
  "  basic\t\t\tPrint simple statistics\n"
  "Mode-dependent types are:\n"
  "  fibers\t\tAnalyze fiber statistics\n"
  "  hparts\t\tAnalyze a hypergraph partitioning\n";

static struct argp_option stats_options[] = {
  { "type", 't', "TYPE", 0, "type of analysis" },
  { "pfile", 'p', "PFILE", 0, "partition file" },
  { 0, 0, 0, 0, "Mode-dependent options:", 1},
  { "mode", 'm', "MODE", 0, "tensor mode to analyze (default: 1)" },
  { 0 }
};

typedef struct
{
  char * ifname;
  char * pfname;
  splatt_stats_type type;
  idx_t mode;
} stats_args;

static error_t parse_stats_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  stats_args *args = state->input;
  switch(key) {
  case 'm':
    args->mode = atoi(arg) - 1;
    break;

  case 't':
    if(strcmp(arg, "basic") == 0) {
      args->type = STATS_BASIC;
    } else if(strcmp(arg, "fibers") == 0) {
      args->type = STATS_FIBERS;
    } else if(strcmp(arg, "hparts") == 0) {
      args->type = STATS_HPARTS;
    } else {
      args->type = STATS_ERROR;
    }
    break;

  case 'p':
    args->pfname = arg;
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

static struct argp stats_argp =
  {stats_options, parse_stats_opt, stats_args_doc, stats_doc};
void splatt_stats(
  int argc,
  char ** argv)
{
  stats_args args;
  args.ifname = NULL;
  args.pfname = NULL;
  args.type = STATS_BASIC;
  args.mode = 0;
  argp_parse(&stats_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  tt_stats(args.ifname, args.type, args.mode, args.pfname);
}



/******************************************************************************
 * SPLATT CONVERT
 *****************************************************************************/
static char convert_args_doc[] = "TENSOR OUTPUT";
static char convert_doc[] =
  "splatt-convert -- convert a tensor\n\n"
  "Mode-dependent conversion types are:\n"
  "  hgraph\t\tHypergraph modeling the sparsity pattern of fibers\n"
  "  fibmat\t\tCSR matrix whose rows are fibers\n"
  "Mode-independent conversion types are:\n"
  "  ijkgraph\t\tTri-partite graph model\n";

typedef struct
{
  char * ifname;
  char * ofname;
  idx_t mode;
  splatt_convert_type type;
} convert_args;

static struct argp_option convert_options[] = {
  { 0, 0, 0, 0, "Mode-independent options:", 2},
  { "type", 't', "TYPE", 0, "type of conversion" },
  { 0, 0, 0, 0, "Mode-dependent options:", 1},
  { "mode", 'm', "MODE", 0, "tensor mode to convert (default: 1)"},
  { 0 }
};

static error_t parse_convert_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  convert_args *args = state->input;
  switch(key) {
  case 'm':
    args->mode = atoi(arg) - 1;
    break;

  case 't':
    if(strcmp(arg, "hgraph") == 0) {
      args->type = CNV_FIB_HGRAPH;
    } else if(strcmp(arg, "ijkgraph") == 0) {
      args->type = CNV_IJK_GRAPH;
    } else if(strcmp(arg, "fibmat") == 0) {
      args->type = CNV_FIB_SPMAT;
    }
    break;

  case ARGP_KEY_ARG:
    switch(state->arg_num) {
    case 0:
      args->ifname = arg;
      break;
    case 1:
      args->ofname = arg;
      break;
    default:
      argp_usage(state);
    }
    break;
  case ARGP_KEY_END:
    if(args->ifname == NULL || args->ofname == NULL) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}

static struct argp convert_argp =
  {convert_options, parse_convert_opt, convert_args_doc, convert_doc};
void splatt_convert(
  int argc,
  char ** argv)
{
  convert_args args;
  args.ifname = NULL;
  args.ofname = NULL;
  args.mode = 0;
  args.type= CNV_ERROR;
  argp_parse(&convert_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  tt_convert(args.ifname, args.ofname, args.mode, args.type);
}



/******************************************************************************
 * SPLATT BENCH
 *****************************************************************************/
static char bench_args_doc[] = "TENSOR";
static char bench_doc[] =
  "splatt-bench -- benchmark MTTKRP algorithms\n\n"
  "Available algorithms are:\n"
  "  splatt\tThe algorithm introduced by splatt\n"
  "  giga\t\tGigaTensor algorithm adapted from the MapReduce paradigm\n"
  "  dfacto\tDFacTo algorithm\n"
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
                                       idx_t const niters,
                                       idx_t const * const threads,
                                       idx_t const nruns)
  = {
    [ALG_SPLATT] = bench_splatt,
    [ALG_GIGA]   = bench_giga,
    [ALG_TTBOX]  = bench_ttbox
  };

typedef struct
{
  char * ifname;
  int which[ALG_NALGS];
  char * algerr;
  idx_t niters;
  idx_t nthreads;
  idx_t rank;
  int scale;
} bench_args;

static struct argp_option bench_options[] = {
  {"alg", 'a', "ALG", 0, "algorithm to benchmark"},
  {"iters", 'i', "NITERS", 0, "number of iterations to use (default: 5)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: 1)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"scale", 's', 0, 0, "scale threads from 1 to NTHREADS (by 2)"},
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
  case 't':
    args->nthreads = atoi(arg);
    break;
  case 'r':
    args->rank = atoi(arg);
    break;
  case 's':
    args->scale = 1;
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


void splatt_bench(
  int argc,
  char ** argv)
{
  bench_args args;
  args.ifname = NULL;
  args.niters = 5;
  args.nthreads = 1;
  args.scale = 0;
  args.rank = 10;
  for(int a=0; a < ALG_NALGS; ++a) {
    args.which[a] = 0;
  }
  argp_parse(&bench_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  if(args.which[ALG_ERR]) {
    fprintf(stderr, "SPLATT: algorithm '%s' is not recognized.\n"
                    "Run with '--help' for assistance.\n", args.algerr);
    exit(EXIT_FAILURE);
  }

  sptensor_t * tt = tt_read(args.ifname);
  matrix_t * mats[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mats[m] = mat_rand(tt->dims[m], args.rank);
  }

  /* create an array of nthreads for scaling */
  idx_t *tsizes;
  idx_t tcount;

  if(args.scale) {
    tcount = 1;
    while((idx_t)(1 << tcount) <= args.nthreads) {
      ++tcount;
    }
    tsizes = (idx_t *) malloc(tcount * sizeof(idx_t));

    for(idx_t t=0; t < tcount; ++t) {
      tsizes[t] = (idx_t) (1 << t);
    }

  } else {
    tcount = 1;
    tsizes = (idx_t *) malloc(1 * sizeof(idx_t));
    tsizes[0] = args.nthreads;
  }

  for(int a=0; a < ALG_NALGS; ++a) {
    if(args.which[a]) {
      bench_funcs[a](tt, mats, args.niters, tsizes, tcount);
    }
  }

  free(tsizes);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mat_free(mats[m]);
  }
  tt_free(tt);
}



/******************************************************************************
 * SPLATT MAIN
 *****************************************************************************/
int main(
  int argc,
  char **argv)
{
  //srand(time(NULL));
  init_timers();
  srand(1);
  splatt_args args;
  /* parse argv[0:1] */
  int nargs = argc > 1 ? 2 : 1;
  argp_parse(&cmd_argp, nargs, argv, ARGP_IN_ORDER, 0, &args);

  printf("****************************************************************\n");

  switch(args.cmd) {
  case CMD_STATS:
    splatt_stats(argc-1, argv+1);
    break;
  case CMD_CONVERT:
    splatt_convert(argc-1, argv+1);
    break;
  case CMD_CPD:
  case CMD_BENCH:
    splatt_bench(argc-1, argv+1);
    break;
  case CMD_PERM:
    cmd_not_implemented(args.cmd_str);
    break;
  default:
    break;
  }

  report_times();
  printf("****************************************************************\n");
  return EXIT_SUCCESS;
}

