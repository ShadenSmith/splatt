
#include "base.h"

#include "sptensor.h"

sptensor_t * tt_read(
  char * const fname)
{
  sptensor_t * const tt = (sptensor_t*) malloc(sizeof(sptensor_t));

  FILE * fin;
  if((fin = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
    exit(1);
  }

  /* first count nnz in tensor */
  idx_t nnz = 0;
  char * line = NULL;
  ssize_t read;
  size_t len;
  while((read = getline(&line, &len, fin)) != -1) {
    if(read > 1 && line[0] != '#') {
      ++nnz;
    }
  }
  tt->nnz = nnz;

  free(line);
  fclose(fin);
  return tt;
}

void tt_write(
  sptensor_t const * const tt,
  char * const fname)
{
  FILE * fout;
  if(fname == NULL) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      exit(1);
    }
  }

  for(idx_t n=0; n < tt->nnz; ++n) {
    for(idx_t m=0; m < NMODES; ++m) {
      fprintf(fout, SS_IDX " ", tt->ind[m][n]);
    }
    fprintf(fout, SS_VAL "\n", tt->vals[n]);
  }

  if(fname != NULL) {
    fclose(fout);
  }
}

