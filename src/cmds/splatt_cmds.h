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
  "  tucker\t\tCompute the Tucker Decomposition.\n"
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

static inline void cmd_not_implemented(int argc, char ** argv)
{
  print_header();
  printf("SPLATT: command '%s' is not yet implemented.\n", argv[0]);
}


/******************************************************************************
 * SPLATT COMMANDS
 *****************************************************************************/

/* prototypes */
#ifdef SPLATT_USE_MPI
void splatt_mpi_cpd_cmd(int argc, char ** argv);
#else
void splatt_cpd_cmd(int argc, char ** argv);
#endif
void splatt_tucker_cmd(int argc, char ** argv);
void splatt_bench(int argc, char ** argv);
void splatt_check(int argc, char ** argv);
void splatt_convert(int argc, char ** argv);
void splatt_reorder(int argc, char ** argv);
void splatt_stats(int argc, char ** argv);


typedef struct splatt_cmd
{
  char * ch_cmd;
  void (*func)(int, char**);
} splatt_cmd;

static struct splatt_cmd cmds[] =
{
#ifdef SPLATT_USE_MPI
  { "cpd",     splatt_mpi_cpd_cmd },
#else
  { "cpd",     splatt_cpd_cmd },
#endif
  { "tucker",  splatt_tucker_cmd},
  { "bench",   splatt_bench },
  { "check",   splatt_check },
  { "convert", splatt_convert },
  { "reorder", splatt_reorder },
  { "stats",   splatt_stats },
  { "help",    NULL },
  { NULL,      NULL },
};

#endif
