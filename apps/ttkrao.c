
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
  {"alg", 'a', "ALGORITHM", 0, "use ALGORITHM during benchmarking"},
  {"threads", 't', "THREADS", 0, "use THREADS threads during computation"},
  {0}
};

static error_t parse_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  splatt_args *args = state->input;
  switch(key) {
  case ARGP_KEY_ARG:
    if(state->arg_num > 1) {
      argp_usage(state);
      break;
    }
    args->fname = arg;
    break;
  case ARGP_KEY_END:
    if(state->arg_num < 2) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };

int main(int argc, char **argv)
{
  splatt_args args;
  argp_parse(&argp, argc, argv, 0, 0, &args);

  return EXIT_SUCCESS;
}

