
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../convert.h"


/******************************************************************************
 * SPLATT CONVERT
 *****************************************************************************/
static char convert_args_doc[] = "TENSOR OUTPUT";
static char convert_doc[] =
  "splatt-convert -- Convert a tensor to a different form.\n\n"
  "Mode-dependent conversion types are:\n"
  "  fib\t\tHypergraph modeling the sparsity pattern of fibers\n"
  "  nnz\t\tHypergraph modeling the sparsity pattern of nonzeros (fine-grained)\n"
  "  fibmat\t\tCSR matrix whose rows are fibers\n"
  "Mode-independent conversion types are:\n"
  "  graph\t\tTri-partite graph model\n";

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
    if(strcmp(arg, "fib") == 0) {
      args->type = CNV_FIB_HGRAPH;
    } else if(strcmp(arg, "nnz") == 0) {
      args->type = CNV_NNZ_HGRAPH;
    } else if(strcmp(arg, "graph") == 0) {
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


