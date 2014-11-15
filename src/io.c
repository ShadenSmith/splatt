
#include "base.h"

#include "sptensor.h"
#include "spmatrix.h"

#include "timer.h"

sptensor_t * tt_read(
  char * const fname)
{
  sp_timer_t timer;
  timer_start(&timer);

  sptensor_t * tt = (sptensor_t*) malloc(sizeof(sptensor_t));

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

  /* allocate structures */
  for(idx_t m=0; m < NMODES; ++m) {
    tt->ind[m] = (idx_t*) malloc(nnz * sizeof(idx_t));
  }
  tt->vals = (val_t*) malloc(nnz * sizeof(val_t));

  /* fill in tensor data */
  rewind(fin);
  nnz = 0;
  char *ptr = NULL;
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = strtok(line, " \t");
      sscanf(ptr, SS_IDX, &(tt->ind[0][nnz]));
      for(idx_t m=1; m < NMODES; ++m) {
        ptr = strtok(NULL, " \t");
        sscanf(ptr, SS_IDX, &(tt->ind[m][nnz]));
      }
      ptr = strtok(NULL, " \t");
      sscanf(ptr, SS_VAL, &(tt->vals[nnz]));
      ++nnz;
    }
  }

  /* now find the max dimension in each mode */
  for(idx_t m=0; m < NMODES; ++m) {
    idx_t const * const ind = tt->ind[m];
    tt->dims[m] = 0;
    for(idx_t n=0; n < nnz; ++n) {
      if(ind[n] > tt->dims[m]) {
        tt->dims[m] = ind[n];
      }
    }
  }

  free(line);
  fclose(fin);

  timer_stop(&timer);
  printf("time: %0.3fs\n", timer.seconds);
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

void spmat_write(
  spmatrix_t const * const mat,
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

  /* write CSR matrix */
  for(idx_t i=0; i < mat->I; ++i) {
    for(idx_t j=mat->rowptr[i]; j < mat->rowptr[j+1]; ++j) {
      fprintf(fout, SS_IDX " " SS_VAL " ", mat->colind[j], mat->vals[j]);
    }
    fprintf(fout, "\n");
  }

}
