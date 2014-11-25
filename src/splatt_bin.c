
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <argp.h>

#include "../include/splatt.h"
#include "convert.h"
#include "stats.h"

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
  "splatt -- a tensor toolkit\n\n"
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
 * SPLATT MAIN
 *****************************************************************************/
int main(
  int argc,
  char **argv)
{
  srand(time(NULL));
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
  case CMD_PERM:
    cmd_not_implemented(args.cmd_str);
    break;
  default:
    break;
  }

  return EXIT_SUCCESS;
}

