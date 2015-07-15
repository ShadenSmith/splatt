
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <stddef.h>

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
  char * ptr = NULL;

  /* first count nnz in tensor */
  idx_t nnz = 0;
  idx_t nmodes = 0;
  char * line = NULL;
  int64_t read;
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
    fprintf(stderr, "SPLATT ERROR: maximum %"SPLATT_PF_IDX" modes supported. "
                    "Found %"SPLATT_PF_IDX". Please recompile with "
                    "MAX_NMODES=%"SPLATT_PF_IDX".\n",
            MAX_NMODES, nmodes, nmodes);
    return NULL;
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

  return tt;
}



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
int splatt_load(
  char const * const fname,
  splatt_idx_t * nmodes,
  splatt_idx_t ** dims,
  splatt_idx_t * nnz,
  splatt_idx_t *** inds,
  splatt_val_t ** vals)
{
  sptensor_t * tt = tt_read_file(fname);

  *nmodes = tt->nmodes;
  *dims = tt->dims;
  *nnz = tt->nnz;
  *vals = tt->vals;
  *inds = tt->ind;

  free(tt);

  return SPLATT_SUCCESS;
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
    return NULL;
  }

  timer_start(&timers[TIMER_IO]);
  sptensor_t * tt = __tt_read_file(fin);
  timer_stop(&timers[TIMER_IO]);
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
      return;
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
  timer_start(&timers[TIMER_IO]);
  for(idx_t n=0; n < tt->nnz; ++n) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      /* files are 1-indexed instead of 0 */
      fprintf(fout, "%"SPLATT_PF_IDX" ", tt->ind[m][n] + 1);
    }
    fprintf(fout, "%"SPLATT_PF_VAL"\n", tt->vals[n]);
  }
  timer_stop(&timers[TIMER_IO]);
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
      return;
    }
  }

  hgraph_write_file(hg, fout);

  fclose(fout);
}

void hgraph_write_file(
  hgraph_t const * const hg,
  FILE * fout)
{
  timer_start(&timers[TIMER_IO]);
  /* print header */
  fprintf(fout, "%"SPLATT_PF_IDX" %"SPLATT_PF_IDX, hg->nhedges, hg->nvtxs);
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
  for(idx_t e=0; e < hg->nhedges; ++e) {
    if(hg->hewts != NULL) {
      fprintf(fout, "%"SPLATT_PF_IDX" ", hg->hewts[e]);
    }
    for(idx_t v=hg->eptr[e]; v < hg->eptr[e+1]; ++v) {
      fprintf(fout, "%"SPLATT_PF_IDX" ", hg->eind[v]+1);
    }
    fprintf(fout, "\n");
  }

  /* print vertex weights */
  if(hg->vwts != NULL) {
    for(idx_t v=0; v < hg->nvtxs; ++v) {
      fprintf(fout, "%"SPLATT_PF_IDX"\n", hg->vwts[v]);
    }
  }
  timer_stop(&timers[TIMER_IO]);
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
      return;
    }
  }

  spmat_write_file(mat, fout);

  if(fout != stdout) {
    fclose(fout);
  }
}

void spmat_write_file(
  spmatrix_t const * const mat,
  FILE * fout)
{
  timer_start(&timers[TIMER_IO]);
  /* write CSR matrix */
  for(idx_t i=0; i < mat->I; ++i) {
    for(idx_t j=mat->rowptr[i]; j < mat->rowptr[i+1]; ++j) {
      fprintf(fout, "%"SPLATT_PF_IDX" %"SPLATT_PF_VAL" ", mat->colind[j], mat->vals[j]);
    }
    fprintf(fout, "\n");
  }
  timer_stop(&timers[TIMER_IO]);
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
      return;
    }
  }

  mat_write_file(mat, fout);

  if(fout != stdout) {
    fclose(fout);
  }
}

void mat_write_file(
  matrix_t const * const mat,
  FILE * fout)
{
  timer_start(&timers[TIMER_IO]);
  idx_t const I = mat->I;
  idx_t const J = mat->J;
  val_t const * const vals = mat->vals;

  if(mat->rowmajor) {
    for(idx_t i=0; i < mat->I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        fprintf(fout, "%+0.8e ", vals[j + (i*J)]);
      }
      fprintf(fout, "\n");
    }
  } else {
    for(idx_t i=0; i < mat->I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        fprintf(fout, "%+0.8e ", vals[i + (j*I)]);
      }
      fprintf(fout, "\n");
    }
  }
  timer_stop(&timers[TIMER_IO]);
}


void vec_write(
  val_t const * const vec,
  idx_t const len,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      return;
    }
  }

  vec_write_file(vec, len, fout);

  if(fout != stdout) {
    fclose(fout);
  }
}

void vec_write_file(
  val_t const * const vec,
  idx_t const len,
  FILE * fout)
{
  timer_start(&timers[TIMER_IO]);

  for(idx_t i=0; i < len; ++i) {
    fprintf(fout, "%"SPLATT_PF_VAL"\n", vec[i]);
  }

  timer_stop(&timers[TIMER_IO]);
}


idx_t * part_read(
  char const * const ifname,
  idx_t const nvtxs,
  idx_t * nparts)
{
  FILE * pfile;
  if((pfile = fopen(ifname, "r")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: unable to open '%s'\n", ifname);
    return NULL;
  }

  *nparts = 0;
  idx_t ret;
  idx_t * arr = (idx_t *) malloc(nvtxs * sizeof(idx_t));
  for(idx_t i=0; i < nvtxs; ++i) {
    if((ret = fscanf(pfile, "%"SPLATT_PF_IDX, &(arr[i]))) == 0) {
      fprintf(stderr, "SPLATT ERROR: not enough elements in '%s'\n", ifname);
      free(arr);
      return NULL;
    }
    if(arr[i] > *nparts) {
      *nparts = arr[i];
    }
  }
  fclose(pfile);

  /* increment to adjust for 0-indexing of partition ids */
  *nparts += 1;

  return arr;
}



/******************************************************************************
 * PERMUTATION FUNCTIONS
 *****************************************************************************/
void perm_write(
  idx_t * perm,
  idx_t const dim,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL) {
    fout = stdout;
  } else {
    if((fout = fopen(fname,"w")) == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", fname);
      return;
    }
  }

  perm_write_file(perm, dim, fout);

  if(fname != NULL) {
    fclose(fout);
  }
}

void perm_write_file(
  idx_t * perm,
  idx_t const dim,
  FILE * fout)
{
  for(idx_t i=0; i < dim; ++i) {
    fprintf(fout, "%"SPLATT_PF_IDX"\n", perm[i]);
  }
}
