
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>

#include "../include/splatt.h"


/* argp vars */
char const *argp_program_version = "splatt v0.0";
char const *argp_program_bug_address = "Shaden Smith <shaden@cs.umn.edu>";
static char doc[] = "ttstats -- print statistics about a tensor";
static char args_doc[] = "TENSOR";

typedef struct
{
  char *fname;
} splatt_args;

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

static struct argp argp = { 0, parse_opt, args_doc, doc };

int main(int argc, char **argv)
{
  splatt_args args;
  argp_parse(&argp, argc, argv, 0, 0, &args);

  tt_stats(args.fname);

  return EXIT_SUCCESS;
}

