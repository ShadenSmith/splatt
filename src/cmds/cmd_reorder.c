
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../stats.h"
#include "../reorder.h"



/******************************************************************************
 * SPLATT PERMUTE
 *****************************************************************************/
static char perm_args_doc[] = "TENSOR";
static char perm_doc[] =
  "splatt-reorder -- Reording the indices of a sparse tensor.\n\n"
  "Mode-independent types are:\n"
  "  rand\t\t\tCreate a random permutation of a tensor\n"
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
    if(strcmp(arg, "rand") == 0) {
      args->type = PERM_RAND;
    } else if(strcmp(arg, "graph") == 0) {
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

int splatt_reorder(
  int argc,
  char ** argv)
{
  perm_args args;
  args.ifname = NULL;
  args.pfname = NULL;
  args.ofname = NULL;
  args.type = PERM_ERROR;
  args.mode = 0;
  argp_parse(&perm_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  if(args.type == PERM_ERROR) {
    fprintf(stderr, "SPLATT: '%s' is an unrecognized permutation type.\n",
      args.typestr);
    return SPLATT_ERROR_BADINPUT;
  }

  print_header();

  sptensor_t * tt = tt_read(args.ifname);
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  /* perform permutation */
  permutation_t * perm = tt_perm(tt, args.type, args.mode, args.pfname);

  /* write output */
  if(args.ofname != NULL) {
    tt_write(tt, args.ofname);

    char * fbuf = splatt_malloc(512 * sizeof(*fbuf));;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      sprintf(fbuf, "%s.mode%"SPLATT_PF_IDX".perm", args.ofname, m);
      perm_write(perm->perms[m], tt->dims[m], fbuf);
    }
    splatt_free(fbuf);
  }

  perm_free(perm);
  tt_free(tt);

  return EXIT_SUCCESS;
}


