
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <argp.h>

#include "../include/splatt.h"
#include "convert.h"

/******************************************************************************
 * SPLATT GLOBAL INFO
 *****************************************************************************/
char const *argp_program_version = "splatt v0.0";
char const *argp_program_bug_address = "Shaden Smith <shaden@cs.umn.edu>";


/******************************************************************************
 * SPLATT CMDS
 *****************************************************************************/
typedef enum
{
  CMD_CPD,
  CMD_BENCH,
  CMD_CONVERT,
  CMD_PERM,
  CMD_STATS,
  CMD_HELP,
  CMD_ERROR
} splatt_cmd;

typedef struct
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
static char stats_doc[] = "splatt-stats -- print statistics about a tensor";
static char stats_args_doc[] = "TENSOR";

typedef struct
{
  char * fname;
} stats_args;

static error_t parse_stats_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  stats_args *args = state->input;
  switch(key) {
  case ARGP_KEY_ARG:
    if(args->fname != NULL) {
      argp_usage(state);
      break;
    }
    args->fname = arg;
    break;
  case ARGP_KEY_END:
    if(args->fname == NULL) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}

static struct argp stats_argp = {0, parse_stats_opt, stats_args_doc, stats_doc};
void splatt_stats(
  int argc,
  char ** argv)
{
  stats_args args;
  args.fname = NULL;
  argp_parse(&stats_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  tt_stats(args.fname);
}



/******************************************************************************
 * SPLATT CONVERT
 *****************************************************************************/
static char convert_doc[] = "splatt-convert -- convert a tensor";
static char convert_args_doc[] = "TENSOR OUTPUT";

typedef struct
{
  char * ifname;
  char * ofname;
  splatt_convert_type type;
} convert_args;

static error_t parse_convert_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  convert_args *args = state->input;
  switch(key) {
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
  {0, parse_convert_opt, convert_args_doc, convert_doc};
void splatt_convert(
  int argc,
  char ** argv)
{
  convert_args args;
  args.ifname = NULL;
  args.ofname = NULL;
  argp_parse(&convert_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  tt_convert(args.ifname, args.ofname, args.type);
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

