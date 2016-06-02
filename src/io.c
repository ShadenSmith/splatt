
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
 * FILE TYPES
 *****************************************************************************/
struct ftype
{
  char * extension;
  splatt_file_type type;
};

static struct ftype file_extensions[] = {
  { ".tns", SPLATT_FILE_TEXT_COORD },
  { ".coo", SPLATT_FILE_TEXT_COORD },
  { ".bin", SPLATT_FILE_BIN_COORD  },
  { NULL, 0}
};


splatt_file_type get_file_type(
    char const * const fname)
{
  if(fname == NULL) {
    return SPLATT_FILE_TEXT_COORD;
  }
  /* find last . in filename */
  char const * const suffix = strrchr(fname, '.');
  if(suffix == NULL) {
    goto NOT_FOUND;
  }

  size_t idx = 0;
  do {
    if(strcmp(suffix, file_extensions[idx].extension) == 0) {
      return file_extensions[idx].type;
    }
  } while(file_extensions[++idx].extension != NULL);


  /* default to text coordinate format */
  NOT_FOUND:
  fprintf(stderr, "SPLATT: extension for '%s' not recognized. "
                  "Defaulting to ASCII coordinate form.\n", fname);
  return SPLATT_FILE_TEXT_COORD;
}


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
  idx_t offsets[MAX_NMODES];
  tt_get_dims(fin, &nmodes, &nnz, dims, offsets);

  if(nmodes > MAX_NMODES) {
    fprintf(stderr, "SPLATT ERROR: maximum %"SPLATT_PF_IDX" modes supported. "
                    "Found %"SPLATT_PF_IDX". Please recompile with "
                    "MAX_NMODES=%"SPLATT_PF_IDX".\n",
            (idx_t) MAX_NMODES, nmodes, nmodes);
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
        tt->ind[m][nnz] = strtoull(ptr, &ptr, 10) - offsets[m];
      }
      tt->vals[nnz++] = strtod(ptr, &ptr);
    }
  }

  free(line);

  return tt;
}


/**
* @brief Write a binary header to an input file.
*
* @param fout The file to write to.
* @param tt The tensor to form a header from.
* @param[out] header The header to write.
*/
static void p_write_tt_binary_header(
  FILE * fout,
  sptensor_t const * const tt,
  bin_header * header)
{
  int32_t type = SPLATT_BIN_COORD;
  fwrite(&type, sizeof(type), 1, fout);

  /* now see if all indices fit in 32bit values */
  uint64_t idx = tt->nnz < UINT32_MAX ?  sizeof(uint32_t) : sizeof(uint64_t);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(tt->dims[m] > UINT32_MAX) {
      idx = sizeof(uint64_t);
      break;
    }
  }

  /* now see if every value can exactly be represented as a float */
  uint64_t val = sizeof(float);
  for(idx_t n=0; n < tt->nnz; ++n) {
    float conv = tt->vals[n];
    if((splatt_val_t) conv != tt->vals[n]) {
      val = sizeof(splatt_val_t);
    }
  }

  header->magic = type;
  header->idx_width = idx;
  header->val_width = val;

  fwrite(&idx, sizeof(idx), 1, fout);
  fwrite(&val, sizeof(val), 1, fout);
}



/**
* @brief Read a COORD tensor from a binary file, converting from smaller idx or
*        val precision if necessary.
*
* @param fin The file to read from.
*
* @return The parsed tensor.
*/
static sptensor_t * p_tt_read_binary_file(
  FILE * fin)
{
  bin_header header;
  read_binary_header(fin, &header);

  idx_t nnz = 0;
  idx_t nmodes = 0;
  idx_t dims[MAX_NMODES];

  fill_binary_idx(&nmodes, 1, &header, fin);
  fill_binary_idx(dims, nmodes, &header, fin);
  fill_binary_idx(&nnz, 1, &header, fin);

  if(nmodes > MAX_NMODES) {
    fprintf(stderr, "SPLATT ERROR: maximum %"SPLATT_PF_IDX" modes supported. "
                    "Found %"SPLATT_PF_IDX". Please recompile with "
                    "MAX_NMODES=%"SPLATT_PF_IDX".\n",
            (idx_t) MAX_NMODES, nmodes, nmodes);
    return NULL;
  }

  /* allocate structures */
  sptensor_t * tt = tt_alloc(nnz, nmodes);
  memcpy(tt->dims, dims, nmodes * sizeof(*dims));

  /* fill in tensor data */
  for(idx_t m=0; m < nmodes; ++m) {
    fill_binary_idx(tt->ind[m], nnz, &header, fin);
  }
  fill_binary_val(tt->vals, nnz, &header, fin);

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

  sptensor_t * tt = NULL;

  timer_start(&timers[TIMER_IO]);
  switch(get_file_type(fname)) {
    case SPLATT_FILE_TEXT_COORD:
      tt = p_tt_read_file(fin);
      break;
    case SPLATT_FILE_BIN_COORD:
      tt = p_tt_read_binary_file(fin);
      break;
  }
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
    idx_t * outdims,
    idx_t * offset)
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
    offset[m] = 1;
  }

  /* fill in tensor dimensions */
  rewind(fin);
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10);

        /* outdim is maximum */
        outdims[m] = (ind > outdims[m]) ? ind : outdims[m];

        /* offset is minimum */
        offset[m] = (ind < offset[m]) ? ind : offset[m];
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
      ++nnz;
    }
  }
  *outnnz = nnz;
  *outnmodes = nmodes;

  /* only support 0 or 1 indexing */
  for(idx_t m=0; m < nmodes; ++m) {
    if(offset[m] != 0 && offset[m] != 1) {
      fprintf(stderr, "SPLATT: ERROR tensors must be 0 or 1 indexed. "
                      "Mode %"SPLATT_PF_IDX" is %"SPLATT_PF_IDX" indexed.\n",
          m, offset[m]);
      exit(1);
    }
  }

  /* adjust dims when zero-indexing */
  for(idx_t m=0; m < nmodes; ++m) {
    if(offset[m] == 0) {
      ++outdims[m];
    }
  }

  rewind(fin);
  free(line);
}


void tt_write(
  sptensor_t const * const tt,
  char const * const fname)
{
  FILE * fout;
  if(fname == NULL) {
    fout = stdout;
  } else if((fout = fopen(fname, "w")) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n", fname);
    return;
  }

  switch(get_file_type(fname)) {
  case SPLATT_FILE_BIN_COORD:
    tt_write_binary_file(tt, fout);
    break;

  case SPLATT_FILE_TEXT_COORD:
  default:
    tt_write_file(tt, fout);
    break;
  }
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

  bin_header header;
  p_write_tt_binary_header(fout, tt, &header);

  /* WRITE INDICES */

  /* if we are writing to the same precision they are stored in, just fwrite */
  if(header.idx_width == sizeof(splatt_idx_t)) {
    fwrite(&tt->nmodes, sizeof(tt->nmodes), 1, fout);
    fwrite(tt->dims, sizeof(*tt->dims), tt->nmodes, fout);
    fwrite(&tt->nnz, sizeof(tt->nnz), 1, fout);
    for(idx_t m=0; m < tt->nmodes; ++m) {
      fwrite(tt->ind[m], sizeof(*tt->ind[m]), tt->nnz, fout);
    }

  /* otherwise we convert (downwards) element-wise */
  } else if(header.idx_width < sizeof(splatt_idx_t)) {
    uint32_t buf = tt->nmodes;
    fwrite(&buf, sizeof(buf), 1, fout);
    for(idx_t m=0; m < tt->nmodes; ++m) {
      buf = tt->dims[m];
      fwrite(&buf, sizeof(buf), 1, fout);
    }
    buf = tt->nnz;
    fwrite(&buf, sizeof(buf), 1, fout);
    /* write inds */
    for(idx_t m=0; m < tt->nmodes; ++m) {
      for(idx_t n=0; n < tt->nnz; ++n) {
        buf = tt->ind[m][n];
        fwrite(&buf, sizeof(buf), 1, fout);
      }
    }

  } else {
    /* XXX this should never be reached */
    fprintf(stderr, "SPLATT: the impossible happened, "
                    "idx_width > IDX_TYPEWIDTH.\n");
    abort();
  }

  /* WRITE VALUES */

  if(header.val_width == sizeof(splatt_val_t)) {
    fwrite(tt->vals, sizeof(*tt->vals), tt->nnz, fout);
  /* otherwise we convert (downwards) element-wise */
  } else if(header.val_width < sizeof(splatt_val_t)) {
    for(idx_t n=0; n < tt->nnz; ++n) {
      float buf = tt->vals[n];
      fwrite(&buf, sizeof(buf), 1, fout);
    }

  } else {
    /* XXX this should never be reached */
    fprintf(stderr, "SPLATT: the impossible happened, "
                    "val_width > VAL_TYPEWIDTH.\n");
    abort();
  }

  timer_stop(&timers[TIMER_IO]);
}


void read_binary_header(
  FILE * fin,
  bin_header * header)
{
  fread(&(header->magic), sizeof(header->magic), 1, fin);
  fread(&(header->idx_width), sizeof(header->idx_width), 1, fin);
  fread(&(header->val_width), sizeof(header->val_width), 1, fin);

  if(header->idx_width > SPLATT_IDX_TYPEWIDTH / 8) {
    fprintf(stderr, "SPLATT: ERROR input has %zu-bit integers. "
                    "Build with SPLATT_IDX_TYPEWIDTH %zu\n",
                    header->idx_width * 8, header->idx_width * 8);
    exit(EXIT_FAILURE);
  }

  if(header->val_width > SPLATT_VAL_TYPEWIDTH / 8) {
    fprintf(stderr, "SPLATT: WARNING input has %zu-bit floating-point values. "
                    "Build with SPLATT_VAL_TYPEWIDTH %zu for full precision\n",
                    header->val_width * 8, header->val_width * 8);
  }
}


void fill_binary_idx(
    idx_t * const buffer,
    idx_t const count,
    bin_header const * const header,
    FILE * fin)
{
  if(header->idx_width == sizeof(splatt_idx_t)) {
    fread(buffer, sizeof(idx_t), count, fin);
  } else {
    uint32_t ubuf;
    for(idx_t n=0; n < count; ++n) {
      fread(&ubuf, sizeof(ubuf), 1, fin);
      buffer[n] = ubuf;
    }
  }
}


void fill_binary_val(
    val_t * const buffer,
    idx_t const count,
    bin_header const * const header,
    FILE * fin)
{
  if(header->val_width == sizeof(splatt_val_t)) {
    fread(buffer, sizeof(val_t), count, fin);
  } else {
    float fbuf;
    for(idx_t n=0; n < count; ++n) {
      fread(&fbuf, sizeof(fbuf), 1, fin);
      buffer[n] = fbuf;
    }
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
        fprintf(fout, "%+0.8le ", vals[j + (i*J)]);
      }
      fprintf(fout, "\n");
    }
  } else {
    for(idx_t i=0; i < mat->I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        fprintf(fout, "%+0.8le ", vals[i + (j*I)]);
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
    fprintf(fout, "%le\n", vec[i]);
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
