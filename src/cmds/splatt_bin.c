
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../timer.h"
#include "../util.h"

#ifdef SPLATT_USE_MPI
#include <mpi.h>
#endif


/******************************************************************************
 * SPLATT GLOBAL INFO
 *****************************************************************************/
char const *argp_program_version = "splatt " \
  "v"SPLATT_STRFY(SPLATT_VER_MAJOR) \
  "."SPLATT_STRFY(SPLATT_VER_MINOR) \
  "."SPLATT_STRFY(SPLATT_VER_SUBMINOR);

char const *argp_program_bug_address = "Shaden Smith <shaden@cs.umn.edu>";


/******************************************************************************
 * SPLATT CMDS
 *****************************************************************************/

typedef struct splatt_args
{
  char * cmd_str;
  splatt_cmd cmd;
} splatt_args;

static splatt_cmd read_cmd(char const * const str)
{
  splatt_cmd cmd = CMD_ERROR;
  if(strcmp(str, "cpd") == 0) {
    cmd = CMD_CPD;
  } else if(strcmp(str, "bench") == 0) {
    cmd = CMD_BENCH;
  } else if(strcmp(str, "check") == 0) {
    cmd = CMD_CHECK;
  } else if(strcmp(str, "convert") == 0) {
    cmd = CMD_CONVERT;
  } else if(strcmp(str, "reorder") == 0) {
    cmd = CMD_REORDER;
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

static inline void print_footer(void)
{
  report_times();
  printf("****************************************************************\n");
}


/******************************************************************************
 * SPLATT MAIN
 *****************************************************************************/

int main(
  int argc,
  char **argv)
{
  int rank = 0;
#ifdef SPLATT_USE_MPI
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

  srand(time(NULL) * (rank+1));

  /* initialize timers */
  init_timers();
  timer_start(&timers[TIMER_ALL]);

  /* parse argv[0:1] */
  splatt_args args;
  int nargs = argc > 1 ? 2 : 1;
  argp_parse(&cmd_argp, nargs, argv, ARGP_IN_ORDER, 0, &args);

  /* execute the cmd! */
  splatt_cmds[args.cmd](argc-1, argv+1);

#ifdef SPLATT_USE_MPI
  /* all done */
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  timer_stop(&timers[TIMER_ALL]);
  if(rank == 0) {
    print_footer();
  }

#ifdef SPLATT_USE_MPI
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}

