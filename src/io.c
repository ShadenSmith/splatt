
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
static sptensor_t * p_tt_read_file(
  FILE * fin)
{
  char * ptr = NULL;

  /* first count nnz in tensor */
  idx_t nnz = 0;
  idx_t nmodes = 0;

  idx_t dims[MAX_NMODES];
  tt_get_dims(fin, &nmodes, &nnz, dims);

  if(nmodes > MAX_NMODES) {
    fprintf(stderr, "SPLATT ERROR: maximum %"SPLATT_PF_IDX" modes supported. "
                    "Found %"SPLATT_PF_IDX". Please recompile with "
                    "MAX_NMODES=%"SPLATT_PF_IDX".\n",
            MAX_NMODES, nmodes, nmodes);
    return NULL;
  }

  /* allocate structures */
  sptensor_t * tt = tt_alloc(nnz, nmodes);
  memcpy(tt->dims, dims, nmodes * sizeof(*dims));

  char * line = NULL;
  int64_t read;
  size_t len = 0;

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

  free(line);

  return tt;
}


static sptensor_t * p_tt_read_binary_file(
  FILE * fin)
{
  char * ptr = NULL;

  /* first count nnz in tensor */
  idx_t nnz = 0;
  idx_t nmodes = 0;

  idx_t dims[MAX_NMODES];
  tt_get_dims_binary(fin, &nmodes, &nnz, dims);

  if(nmodes > MAX_NMODES) {
    fprintf(stderr, "SPLATT ERROR: maximum %"SPLATT_PF_IDX" modes supported. "
                    "Found %"SPLATT_PF_IDX". Please recompile with "
                    "MAX_NMODES=%"SPLATT_PF_IDX".\n",
            MAX_NMODES, nmodes, nmodes);
    return NULL;
  }

  /* allocate structures */
  sptensor_t * tt = tt_alloc(nnz, nmodes);
  memcpy(tt->dims, dims, nmodes * sizeof(*dims));

  /* fill in tensor data */
  for (idx_t m=0; m < nmodes; ++m) {
    fread(tt->ind[m], sizeof(*tt->ind[m]), nnz, fin);
  }
  fread(tt->vals, sizeof(*tt->vals), nnz, fin);

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
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

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
  sptensor_t * tt = p_tt_read_file(fin);
  timer_stop(&timers[TIMER_IO]);
  fclose(fin);
  return tt;
}


sptensor_t * tt_read_binary_file(
  char const * const fname)
{
  FILE * fin;
  if((fin = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n", fname);
    return NULL;
  }

  timer_start(&timers[TIMER_IO]);
  sptensor_t * tt = p_tt_read_binary_file(fin);
  timer_stop(&timers[TIMER_IO]);
  fclose(fin);
  return tt;
}


void tt_get_dims(
    FILE * fin,
    idx_t * const outnmodes,
    idx_t * const outnnz,
    idx_t * outdims)
{
  char * ptr = NULL;
  idx_t nnz = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  /* first count modes in tensor */
  idx_t nmodes = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    if(read > 1 && line[0] != '#') {
      /* get nmodes from first nnz line */
      ptr = strtok(line, " \t");
      while(ptr != NULL) {
        ++nmodes;
        ptr = strtok(NULL, " \t");
      }
      break;
    }
  }
  --nmodes;

  for(idx_t m=0; m < nmodes; ++m) {
    outdims[m] = 0;
  }

  /* fill in tensor dimensions */
  rewind(fin);
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10);
        outdims[m] = (ind > outdims[m]) ? ind : outdims[m];
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
      ++nnz;
    }
  }
  *outnnz = nnz;
  *outnmodes = nmodes;

  rewind(fin);
  free(line);
}


void tt_get_dims_binary(
    FILE * fin,
    idx_t * const outnmodes,
    idx_t * const outnnz,
    idx_t * outdims)
{
  char * ptr = NULL;
  idx_t nnz = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  fread(outnmodes, sizeof(*outnmodes), 1, fin);
  fread(outdims, sizeof(*outdims), *outnmodes, fin);
  fread(outnnz, sizeof(*outnnz), 1, fin);
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


void tt_write_binary(
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

  tt_write_binary_file(tt, fout);

  if(fname != NULL) {
    fclose(fout);
  }
}


void tt_write_binary_file(
  sptensor_t const * const tt,
  FILE * fout)
{
  timer_start(&timers[TIMER_IO]);

  fwrite(&tt->nmodes, sizeof(tt->nmodes), 1, fout);
  fwrite(tt->dims, sizeof(*tt->dims), tt->nmodes, fout);
  fwrite(&tt->nnz, sizeof(tt->nnz), 1, fout);

  for(idx_t m=0; m < tt->nmodes; ++m) {
    fwrite(tt->ind[m], sizeof(*tt->ind[m]), tt->nnz, fout);
  }
  fwrite(tt->vals, sizeof(*tt->vals), tt->nnz, fout);

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



void graph_write_file(
    splatt_graph const * const graph,
    FILE * fout)
{
  timer_start(&timers[TIMER_IO]);
  /* print header */
  fprintf(fout, "%"SPLATT_PF_IDX" %"SPLATT_PF_IDX" 0%d%d", graph->nvtxs,
      graph->nedges/2, graph->vwgts != NULL, graph->ewgts != NULL);
  /* handle multi-constraint partitioning */
  if(graph->nvwgts > 1) {
    fprintf(fout, " %"SPLATT_PF_IDX, graph->nvwgts);
  }
  fprintf(fout, "\n");

  /* now write adj list */
  for(vtx_t v=0; v < graph->nvtxs; ++v) {
    /* vertex weights */
    if(graph->vwgts != NULL) {
      for(idx_t x=0; x < graph->nvwgts; ++x) {
        fprintf(fout, "%"SPLATT_PF_IDX" ", graph->vwgts[x+(v*graph->nvwgts)]);
      }
    }

    for(adj_t e=graph->eptr[v]; e < graph->eptr[v+1]; ++e) {
      fprintf(fout, "%"SPLATT_PF_IDX" ", graph->eind[e] + 1);
      /* edge weight */
      if(graph->ewgts != NULL) {
        fprintf(fout, "%"SPLATT_PF_IDX" ", graph->ewgts[e]);
      }
    }
    fprintf(fout, "\n");
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
  idx_t * arr = (idx_t *) splatt_malloc(nvtxs * sizeof(idx_t));
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
