
#include "base.h"

#include "sptensor.h"

sptensor_t * read_sptensor(
  char * const fname)
{
  sptensor_t * const tt = (sptensor_t*) malloc(sizeof(sptensor_t));

  FILE * fin;
  if((fin = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
    exit(1);
  }

  fclose(fin);
  return tt;
}

