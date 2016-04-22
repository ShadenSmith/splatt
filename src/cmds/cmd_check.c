
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../sptensor.h"
#include "../io.h"


/******************************************************************************
 * SPLATT CHECK
 *****************************************************************************/
static char check_args_doc[] = "TENSOR";
static char check_doc[] =
  "splatt-check -- check a tensor file for correctness.\n\n"
  "Checks for:\n"
  "  duplicate nonzeros (fixed via repeated averaging)\n"
  "  empty slices (fixed via mode<m>.map file)\n";

static struct argp_option check_options[] = {
  { "fix", 'f', "FILE", OPTION_ARG_OPTIONAL, "fix mistakes and write to FILE" },
  { 0 }
};

typedef struct
{
  char * ifname;
  char * ofname;
  int fix;
} check_args;

static error_t parse_check_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  check_args *args = state->input;
  switch(key) {
  case 'f':
    args->fix = 1;
    args->ofname = arg;
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

static struct argp check_argp =
  {check_options, parse_check_opt, check_args_doc, check_doc};

int splatt_check(
  int argc,
  char ** argv)
{
  check_args args;
  args.ifname = NULL;
  args.ofname = NULL;
  args.fix = 0;
  argp_parse(&check_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  print_header();
  sptensor_t * tt = tt_read(args.ifname);
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  idx_t const rnnz = tt_remove_dups(tt);
  idx_t const rslices = tt_remove_empty(tt);

  if(rnnz == 0 && rslices == 0) {
    printf("NO ERRORS FOUND.\n");
  } else {
    printf("%"SPLATT_PF_IDX " DUPLICATES FOUND.\n", rnnz);
    printf("%"SPLATT_PF_IDX " EMPTY SLICES FOUND.\n", rslices);

    if(args.fix == 1) {
      /* write fixed tensor */
      tt_write(tt, args.ofname);

      /* write maps to file if present */
      for(idx_t m=0; m < tt->nmodes; ++m) {
        idx_t const * const map = tt->indmap[m];
        if(map != NULL) {
          char * buf = NULL;
          if(asprintf(&buf, "mode%"SPLATT_PF_IDX".map", m+1) == -1) {
            fprintf(stderr, "SPLATT: asprintf failed\n");
            abort();
          }
          FILE * fout = fopen(buf, "w");
          for(idx_t i=0; i < tt->dims[m]; ++i) {
            fprintf(fout, "%"SPLATT_PF_IDX"\n", 1+map[i]);
          }
          fclose(fout);
          free(buf);
        }
      }
    } /* if fix */
  } /* if errors */

  tt_free(tt);

  return EXIT_SUCCESS;
}

