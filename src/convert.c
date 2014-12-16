
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"
#include "ftensor.h"
#include "graph.h"
#include "io.h"
#include "matrix.h"
#include "convert.h"


/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/
static void __convert_fib_hgraph(
  sptensor_t * tt,
  idx_t const mode,
  char const * const ofname)
{
  ftensor_t * ft = ften_alloc(tt, 0);

  hgraph_t * hg = hgraph_fib_alloc(ft, mode);
  hgraph_write(hg, ofname);

  hgraph_free(hg);
  ften_free(ft);
}

static void __convert_fib_mat(
  sptensor_t * tt,
  idx_t const mode,
  char const * const ofname)
{
  ftensor_t * ft = ften_alloc(tt, 0);
  spmatrix_t * mat = ften_spmat(ft, mode);

  spmat_write(mat, ofname);

  spmat_free(mat);
  ften_free(ft);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_convert(
  char const * const ifname,
  char const * const ofname,
  idx_t const mode,
  splatt_convert_type const type)
{
  sptensor_t * tt = tt_read(ifname);

  switch(type) {
  case CNV_FIB_HGRAPH:
    __convert_fib_hgraph(tt, mode, ofname);
    break;
  case CNV_FIB_SPMAT:
    __convert_fib_mat(tt, mode, ofname);
    break;
  default:
    fprintf(stderr, "SPLATT ERROR: convert type not implemented.\n");
    exit(1);
  }

  tt_free(tt);
}

