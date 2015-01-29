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
  "  cpd\t\tCompute the Canonical Polyadic Decomposition\n"
  "  bench\t\tBenchmark MTTKRP algorithms\n"
  "  convert\tConvert a tensor to different formats\n"
  "  reorder\t\tReorder a tensor using one of several methods\n"
  "  stats\t\tPrint tensor statistics\n"
  "  help\t\tPrint this help message\n";

static inline void print_header(void)
{
  printf("****************************************************************\n");
  printf("splatt built from %s\n\n", VERSION_STR);
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
void splatt_cpd(int argc, char ** argv);
void splatt_bench(int argc, char ** argv);
void splatt_convert(int argc, char ** argv);
void splatt_reorder(int argc, char ** argv);
void splatt_stats(int argc, char ** argv);


typedef enum splatt_cmd
{
  CMD_CPD,
  CMD_BENCH,
  CMD_CONVERT,
  CMD_REORDER,
  CMD_STATS,
  CMD_HELP,
  CMD_ERROR,
  CMD_NCMDS
} splatt_cmd;


static void (*splatt_cmds[CMD_NCMDS]) (int argc, char ** argv) = {
  [CMD_CPD]     = splatt_cpd,
  [CMD_BENCH]   = splatt_bench,
  [CMD_CONVERT] = splatt_convert,
  [CMD_REORDER] = splatt_reorder,
  [CMD_STATS]   = splatt_stats,
};

#endif
