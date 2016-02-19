#ifndef SPLATT_CMDS_H
#define SPLATT_CMDS_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../base.h"
#include <argp.h>


/******************************************************************************
 * PRIVATE STRUCTURES
 *****************************************************************************/
static char cmd_args_doc[] = "CMD";
static char cmd_doc[] =
  "splatt -- the Surprisingly ParalleL spArse Tensor Toolkit\n\n"
  "The available commands are:\n"
  "  cpd\t\tCompute the Canonical Polyadic Decomposition.\n"
  "  complete\tComplete a tensor with missing entries.\n"
  "  bench\t\tBenchmark MTTKRP algorithms.\n"
  "  check\t\tCheck a tensor file for correctness.\n"
  "  convert\tConvert a tensor to different formats.\n"
  "  reorder\t\tReorder a tensor using one of several methods.\n"
  "  stats\t\tPrint tensor statistics.\n"
  "  help\t\tPrint this help message.\n";


/**
* @brief Print header for splatt executable. Prints version information and
*        Git build information, if available.
*/
static inline void print_header(void)
{
  printf("****************************************************************\n");
#ifdef SPLATT_VERSION_STR
  /* include git build info if available */
  printf("splatt v%d.%d.%d built from %s\n\n",
      SPLATT_VER_MAJOR, SPLATT_VER_MINOR, SPLATT_VER_SUBMINOR,
      SPLATT_VERSION_STR);
#else
  printf("splatt v%d.%d.%d\n\n",
      SPLATT_VER_MAJOR, SPLATT_VER_MINOR, SPLATT_VER_SUBMINOR);
#endif
}



/******************************************************************************
 * SPLATT COMMAND PROTOTYPES
 *****************************************************************************/
#ifdef SPLATT_USE_MPI
int splatt_mpi_cpd_cmd(int argc, char ** argv);
#else
int splatt_cpd_cmd(int argc, char ** argv);
#endif
int splatt_bench(int argc, char ** argv);
int splatt_check(int argc, char ** argv);
int splatt_convert(int argc, char ** argv);
int splatt_reorder(int argc, char ** argv);
int splatt_stats(int argc, char ** argv);

int splatt_tc_cmd(int argc, char ** argv);



/******************************************************************************
 * SPLATT COMMAND STRINGS
 *****************************************************************************/
/* typedef function pointer, called splatt_cmd_func */
typedef int (* splatt_cmd_func)(int argc, char ** argv);
typedef struct
{
  char * cmd_str;
  splatt_cmd_func func;
} cmd_struct;


static cmd_struct const splatt_cmds[] = {
#ifdef SPLATT_USE_MPI
  { "cpd", splatt_mpi_cpd_cmd },
#else
  { "cpd", splatt_cpd_cmd },
#endif

  { "complete", splatt_tc_cmd },
  { "bench", splatt_bench },
  { "check", splatt_check },
  { "convert", splatt_convert },
  { "reorder", splatt_reorder },
  { "stats", splatt_stats },
  { "help", NULL},

  { NULL, NULL }
};

#endif
