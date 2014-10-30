
#include <stdio.h>
#include <stdlib.h>

#include "../include/splatt.h"

int main(int argc, char **argv)
{
  if(argc == 1) {
    fprintf(stderr, "usage: %s <tensor>\n", argv[0]);
    return EXIT_FAILURE;
  }

  tt_stats(argv[1]);

  return EXIT_SUCCESS;
}
