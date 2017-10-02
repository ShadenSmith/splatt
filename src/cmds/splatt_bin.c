
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

static splatt_cmd_func read_cmd(char const * const str)
{
  int x = 0;
  while(splatt_cmds[x].cmd_str != NULL) {
    if(strcmp(str, splatt_cmds[x].cmd_str) == 0) {
      return splatt_cmds[x].func;
    }
    ++x;
  }
  return NULL;
}



static error_t parse_cmd(
  int key,
  char * arg,
  struct argp_state * state)
{
  cmd_struct * args = state->input;
  switch(key) {
  case ARGP_KEY_ARG:
    if(state->arg_num == 0) {
      args->cmd_str = arg;
      args->func = read_cmd(arg);
      if(args->func == NULL) {
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
 * SPLATT MAIN
 *****************************************************************************/

int main(
  int argc,
  char **argv)
{
  setvbuf(stdout, NULL, _IOLBF, 0);

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
  cmd_struct args;
  int nargs = argc > 1 ? 2 : 1;
  argp_parse(&cmd_argp, nargs, argv, ARGP_IN_ORDER, 0, &args);

  /* execute the cmd! */
  int ret = args.func(argc-1, argv+1);

#ifdef SPLATT_USE_MPI
  /* all done */
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  timer_stop(&timers[TIMER_ALL]);
  if(rank == 0) {
    report_times();
    printf("****************************************************************\n");
  }

#ifdef SPLATT_USE_MPI
  MPI_Finalize();
#endif

  return ret;
}

