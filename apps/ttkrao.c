
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>

#include "../include/splatt.h"

char const *argp_program_version = "splatt v0.0";
char const *argp_program_bug_address = "Shaden Smith <shaden@cs.umn.edu>";
static char doc[] = "ttkrao -- benchmark Tensor Times Khatri-Rao product algorithms";
static char args_doc[] = "TENSOR";

typedef struct
{
  char *fname;
} splatt_args;

static struct argp_option options[] = {
  {"alg", 'a', "ALGORITHM", 0, "use ALG"},
  {"threads", 't', "THREADS", 0, "use THREADS threads during computation"},
  {0}
};

int main(int argc, char **argv)
{
  printf("ttkrao\n");

  return EXIT_SUCCESS;
}
