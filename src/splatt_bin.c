
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <argp.h>

#include "../include/splatt.h"

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
  CMD_STATS,
  CMD_PERM,
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
  "  stats\t\tPrint tensor statistics\n"
  "  perm\t\tPermute a tensor using one of several methods\n"
  "  help\t\tPrint this help message\n";

void cmd_not_implemented(char const * const cmd)
{
  printf("SPLATT: option '%s' is not implemented.\n", cmd);
  exit(1);
}

static splatt_cmd read_cmd(char const * const str)
{
  splatt_cmd cmd = CMD_ERROR;
  if(strcmp(str, "cpd") == 0) {
    cmd = CMD_CPD;
  } else if(strcmp(str, "bench") == 0) {
    cmd = CMD_BENCH;
  } else if(strcmp(str, "stats") == 0) {
    cmd = CMD_STATS;
  } else if(strcmp(str, "perm") == 0) {
    cmd = CMD_PERM;
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

int main(
  int argc,
  char **argv)
{
  splatt_args args;
  /* parse argv[0:1] */
  int nargs = argc > 1 ? 2 : 1;
  argp_parse(&cmd_argp, nargs, argv, ARGP_IN_ORDER, 0, &args);

  switch(args.cmd) {
  case CMD_STATS:
    splatt_stats(argc-1, argv+1);
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

