
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../stats.h"


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

