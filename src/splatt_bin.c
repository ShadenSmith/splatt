
#include "base.h"

#include <argp.h>

/* SPLATT MODULES */
#include "timer.h"
#include "io.h"
#include "convert.h"
#include "stats.h"
#include "bench.h"
#include "reorder.h"

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


static inline void print_header(void)
{
  printf("****************************************************************\n");
  printf("splatt built from %s\n\n", VERSION_STR);
}


static inline void print_footer(void)
{
  report_times();
  printf("****************************************************************\n");
}

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

  print_header();

  sptensor_t * tt = tt_read(args.ifname);
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);
  if(args.type != STATS_BASIC) {
    stats_tt(tt, args.ifname, args.type, args.mode, args.pfname);
  }
  tt_free(tt);
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

  print_header();

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
} bench_args;

static struct argp_option bench_options[] = {
  {"alg", 'a', "ALG", 0, "algorithm to benchmark"},
  {"iters", 'i', "NITERS", 0, "number of iterations to use (default: 5)"},
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

  if(args.pfname != NULL) {
    printf("Reordering ------------------------------------------------------\n");

  }

  printf("Benchmarking ---------------------------------------------------\n");

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

  /* fill bench opts */
  bench_opts opts;
  opts.niters = args.niters;
  opts.threads = tsizes;
  opts.nruns = tcount;
  opts.write = args.write;

  /* initialize perms */
  for(idx_t m=0; m < tt->nmodes; ++m ){
    opts.perm.perms[m] = NULL;
    opts.perm.iperms[m] = NULL;
  }

  for(int a=0; a < ALG_NALGS; ++a) {
    if(args.which[a]) {
      bench_funcs[a](tt, mats, &opts);
    }
  }

  free(tsizes);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mat_free(mats[m]);
  }
  mat_free(mats[MAX_NMODES]);
  tt_free(tt);
}



/******************************************************************************
 * SPLATT PERMUTE
 *****************************************************************************/
static char perm_args_doc[] = "TENSOR";
static char perm_doc[] =
  "splatt-perm -- permute a tensor\n\n"
  "Mode-independent types are:\n"
  "  graph\t\t\tReorder based on the partitioning of a mode-independent graph\n"
  "Mode-dependent types are:\n"
  "  hgraph\t\tReorder based on the partitioning of a fiber hyper-graph\n";

static struct argp_option perm_options[] = {
  { "type", 't', "TYPE", 0, "type of reordering" },
  { "pfile", 'p', "PFILE", 0, "partition file" },
  { "outfile", 'o', "FILE", 0, "write reordered tensor to file" },
  { 0, 0, 0, 0, "Mode-dependent options:", 1},
  { "mode", 'm', "MODE", 0, "tensor mode to analyze (default: 1)" },
  { 0 }
};

typedef struct
{
  char * ifname;
  char * ofname;
  char * pfname;
  char * typestr;
  splatt_perm_type type;
  idx_t mode;
} perm_args;

static error_t parse_perm_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  perm_args *args = state->input;
  switch(key) {
  case 'm':
    args->mode = atoi(arg) - 1;
    break;

  case 'o':
    args->ofname = arg;
    break;

  case 't':
    if(strcmp(arg, "graph") == 0) {
      args->type = PERM_GRAPH;
    } else if(strcmp(arg, "hgraph") == 0) {
      args->type = PERM_HGRAPH;
    } else {
      args->typestr = arg;
      args->type = PERM_ERROR;
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

static struct argp perm_argp =
  {perm_options, parse_perm_opt, perm_args_doc, perm_doc};
void splatt_perm(
  int argc,
  char ** argv)
{
  perm_args args;
  args.ifname = NULL;
  args.pfname = NULL;
  args.type = PERM_ERROR;
  args.mode = 0;
  argp_parse(&perm_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  if(args.type == PERM_ERROR) {
    fprintf(stderr, "SPLATT: '%s' is an unrecognized permutation type.\n",
      args.typestr);
    exit(EXIT_FAILURE);
  }

  print_header();

  sptensor_t * tt = tt_read(args.ifname);
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);
  tt_perm(tt, args.type, args.mode, args.pfname);
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
  timer_start(&timers[TIMER_ALL]);
  srand(1);
  splatt_args args;
  /* parse argv[0:1] */
  int nargs = argc > 1 ? 2 : 1;
  argp_parse(&cmd_argp, nargs, argv, ARGP_IN_ORDER, 0, &args);

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
    splatt_perm(argc-1, argv+1);
    break;
  default:
    break;
  }

  timer_stop(&timers[TIMER_ALL]);
  print_footer();

  return EXIT_SUCCESS;
}

