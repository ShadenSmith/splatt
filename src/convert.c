
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"
#include "ftensor.h"
#include "graph.h"
#include "io.h"
#include "convert.h"


/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/
static void __convert_ijk_graph(
  sptensor_t * tt,
  char const * const ofname)
{

}

static void __convert_fib_hgraph(
  sptensor_t * tt,
  idx_t const mode,
  char const * const ofname)
{
  ftensor_t * ft = ften_alloc(tt);

  hgraph_t * hg = hgraph_fib_alloc(ft, mode);
  hgraph_write(hg, ofname);

  hgraph_free(hg);
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
  __convert_fib_hgraph(tt, mode, ofname);

  tt_free(tt);
}

