#ifndef SPLATT_IO_H
#define SPLATT_IO_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "matrix.h"
#include "graph.h"
#include "reorder.h"


/**
* @brief Open a file.
*
* @param fname The name of the file.
* @param mode The mode for opening.
*
* @return A FILE pointer.
*/
static inline FILE * open_f(
  char const * const fname,
  char const * const mode)
{
  FILE * f;
  if((f = fopen(fname, mode)) == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n", fname);
    exit(1);
  }
  return f;
}




/******************************************************************************
 * FILE TYPES
 *****************************************************************************/

/**
* @brief An enum describing the supported input file types.
*/
typedef enum
{
  SPLATT_FILE_TEXT_COORD,      /* plain list of tuples + values */
  SPLATT_FILE_BIN_COORD        /* a binary version of the coordinate format */
} splatt_file_type;


#define get_file_type splatt_get_file_type
/**
* @brief Attempt to determine the type of tensor file based on the extension.
*        NOTE: Defaults to SPLATT_FILE_TEXT_COORD and prints to stderr if
*        unable to determine.
*
* @param fname The filename to analyze.
*
* @return A file type.
*/
splatt_file_type get_file_type(
    char const * const fname);



/******************************************************************************
 * BINARY READS
 *****************************************************************************/


typedef enum
{
  SPLATT_BIN_COORD,
  SPLATT_BIN_CSF
} splatt_magic_type;


/**
* @brief This struct is written to the beginning of any binary tensor file
*        written by SPLATT.
*/
typedef struct
{
  int32_t magic;
  uint64_t idx_width;
  uint64_t val_width;
} bin_header;


#define read_binary_header splatt_read_binary_header
/**
* @brief Populate a binary header from an input file.
*
* @param fin The file to read from.
* @param[OUT] header The header to populate.
*/
void read_binary_header(
    FILE * fin,
    bin_header * header);



/******************************************************************************
 * TENSOR FUNCTIONS
 *****************************************************************************/
#define tt_get_dims splatt_tt_get_dims
void tt_get_dims(
    FILE * fin,
    idx_t * const outnmodes,
    idx_t * const outnnz,
    idx_t * outdims);

#define tt_get_dims_binary splatt_tt_get_dims_binary
void tt_get_dims_binary(
    FILE * fin,
    idx_t * const outnmodes,
    idx_t * const outnnz,
    idx_t * outdims);


#define tt_read_file splatt_tt_read_file
sptensor_t * tt_read_file(
  char const * const fname);

#define tt_read_binary_file splatt_tt_read_binary_file
sptensor_t * tt_read_binary_file(
  char const * const fname);

#define tt_write_file splatt_tt_write_file
void tt_write_file(
  sptensor_t const * const tt,
  FILE * fout);

void tt_write_binary_file(
  sptensor_t const * const tt,
  FILE * fout);

#define tt_write splatt_tt_write
void tt_write(
  sptensor_t const * const tt,
  char const * const fname);

#define tt_write_binary_file tt_write_binary_file
void tt_write_binary(
  sptensor_t const * const tt,
  char const * const fname);

/******************************************************************************
 * GRAPH FUNCTIONS
 *****************************************************************************/
#define hgraph_write_file splatt_hgraph_write_file
void hgraph_write_file(
  hgraph_t const * const hg,
  FILE * fout);

#define hgraph_write splatt_hgraph_write
void hgraph_write(
  hgraph_t const * const hg,
  char const * const fname);


#define graph_write_file splatt_graph_write_file
/**
* @brief Write a graph to a file.
*
* @param graph The graph to write.
* @param fout The FILE object to write to.
*/
void graph_write_file(
  splatt_graph const * const graph,
  FILE * fout);


/******************************************************************************
 * DENSE MATRIX FUNCTIONS
 *****************************************************************************/
#define mat_write splatt_mat_write
void mat_write(
  matrix_t const * const mat,
  char const * const fname);

#define mat_write_file splatt_mat_write_file
void mat_write_file(
  matrix_t const * const mat,
  FILE * fout);

#define vec_write splatt_vec_write
void vec_write(
  val_t const * const vec,
  idx_t const len,
  char const * const fname);

#define vec_write_file splatt_vec_write_file
void vec_write_file(
  val_t const * const vec,
  idx_t const len,
  FILE * fout);


/******************************************************************************
 * SPARSE MATRIX FUNCTIONS
 *****************************************************************************/
#define spmat_write splatt_spmat_write
void spmat_write(
  spmatrix_t const * const mat,
  char const * const fname);

#define spmat_write_file splatt_spmat_write_file
void spmat_write_file(
  spmatrix_t const * const mat,
  FILE * fout);


/******************************************************************************
 * PERMUTATION FUNCTIONS
 *****************************************************************************/
#define perm_write splatt_perm_write
void perm_write(
  idx_t * perm,
  idx_t const dim,
  char const * const fname);

#define perm_write_file splatt_perm_write_file
void perm_write_file(
  idx_t * perm,
  idx_t const dim,
  FILE * fout);


/******************************************************************************
 * PARTITION FUNCTIONS
 *****************************************************************************/
#define part_read splatt_part_read
idx_t * part_read(
  char const * const ifname,
  idx_t const nvtxs,
  idx_t * nparts);

#endif
