
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "io.h"
#include "sptensor.h"
#include "matrix.h"
#include "graph.h"

#include "timer.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/
static sptensor_t * __tt_read_file(
  FILE * fin)
{
  sp_timer_t timer;
  timer_reset(&timer);
  timer_start(&timer);

  char * ptr = NULL;

  /* first count nnz in tensor */
  idx_t nnz = 0;
  idx_t nmodes = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    if(read > 1 && line[0] != '#') {
      /* get nmodes from first nnz line */
      if(nnz == 0) {
        ptr = strtok(line, " \t");
        while(ptr != NULL) {
          ++nmodes;
          ptr = strtok(NULL, " \t");
        }
      }
      ++nnz;
    }
  }
  --nmodes;

  if(nmodes > MAX_NMODES) {
    fprintf(stderr, "SPLATT ERROR: maximum " SS_IDX " modes supported. Found "
                    SS_IDX ". Please recompile with MAX_NMODES=" SS_IDX".\n",
            MAX_NMODES, nmodes, nmodes);
    exit(1);
  }

  /* allocate structures */
  sptensor_t * tt = tt_alloc(nnz, nmodes);

  /* fill in tensor data */
  rewind(fin);
  nnz = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        tt->ind[m][nnz] = strtoull(ptr, &ptr, 10) - 1;
      }
      tt->vals[nnz++] = strtod(ptr, &ptr);
    }
  }

  /* now find the max dimension in each mode */
  for(idx_t m=0; m < nmodes; ++m) {
    idx_t const * const ind = tt->ind[m];
    tt->dims[m] = 0;
    for(idx_t n=0; n < nnz; ++n) {
      if(ind[n] > tt->dims[m]) {
        tt->dims[m] = ind[n];
      }
    }
    tt->dims[m] += 1; /* account for 0-indexed ind */
  }

  free(line);

  timer_stop(&timer);
  printf("IO: %0.3fs\n", timer.seconds);
  return tt;
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
sptensor_t * tt_read_file(
  char const * const fname)
{
  FILE * fin;
  if((fin = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n", fname);
    exit(1);
  }

  sptensor_t * tt = __tt_read_file(fin);
  fclose(fin);
  return tt;
}


void tt_write(
  sptensor_t const * const tt,
  char const * const fname)
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

  tt_write_file(tt, fout);

  if(fname != NULL) {
    fclose(fout);
  }
}

void tt_write_file(
  sptensor_t const * const tt,
  FILE * fout)
{
  for(idx_t n=0; n < tt->nnz; ++n) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      fprintf(fout, SS_IDX " ", tt->ind[m][n]);
    }
    fprintf(fout, SS_VAL "\n", tt->vals[n]);
  }
}

void hgraph_write(
  hgraph_t const * const hg,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL || strcmp(fname, "-") == 0) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      exit(1);
    }
  }

  hgraph_write_file(hg, fout);

  fclose(fout);
}

void hgraph_write_file(
  hgraph_t const * const hg,
  FILE * fout)
{
  /* print header */
  fprintf(fout, "%d %d", hg->nhedges, hg->nvtxs);
  if(hg->vwts != NULL) {
    if(hg->hewts != NULL) {
      fprintf(fout, " 11");
    } else {
      fprintf(fout, " 10");
    }
  } else if(hg->hewts != NULL) {
    fprintf(fout, " 1");
  }
  fprintf(fout, "\n");

  /* print hyperedges */
  for(int e=0; e < hg->nhedges; ++e) {
    if(hg->hewts != NULL) {
      fprintf(fout, "%d ", hg->hewts[e]);
    }
    for(int v=hg->eptr[e]; v < hg->eptr[e+1]; ++v) {
      fprintf(fout, "%d ", hg->eind[v]+1);
    }
    fprintf(fout, "\n");
  }

  /* print vertex weights */
  if(hg->vwts != NULL) {
    for(int v=0; v < hg->nvtxs; ++v) {
      fprintf(fout, "%d\n", hg->vwts[v]);
    }
  }
}

void spmat_write(
  spmatrix_t const * const mat,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL || strcmp(fname, "-") == 0) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      exit(1);
    }
  }

  spmat_write_file(mat, fout);

  fclose(fout);
}

void spmat_write_file(
  spmatrix_t const * const mat,
  FILE * fout)
{
  /* write CSR matrix */
  for(idx_t i=0; i < mat->I; ++i) {
    for(idx_t j=mat->rowptr[i]; j < mat->rowptr[i+1]; ++j) {
      fprintf(fout, SS_IDX " " SS_VAL " ", mat->colind[j], mat->vals[j]);
    }
    fprintf(fout, "\n");
  }
}

void mat_write(
  matrix_t const * const mat,
  char const * const fname)
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

  idx_t const I = mat->I;
  idx_t const J = mat->J;
  val_t const * const vals = mat->vals;
  for(idx_t i=0; i < mat->I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      fprintf(fout, SS_VAL " ", vals[j + (i*J)]);
    }
    fprintf(fout, "\n");
  }
}

