
#include "../src/io.h"

#include "ctest/ctest.h"

#include "splatt_test.h"

#include <unistd.h>

static char const * const TMP_FILE = "tmp.txt";


CTEST_DATA(graph)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
};

CTEST_SETUP(graph)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}


CTEST_TEARDOWN(graph)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}


CTEST2(graph, graph_convert)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * const tt = data->tensors[i];

    splatt_graph * graph = graph_convert(tt);

    /* count vtxs */
    vtx_t nv = 0;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      nv += (vtx_t) tt->dims[m];
    }
    ASSERT_EQUAL(nv, graph->nvtxs);

    /* now write graph to tmp.txt and compare against good graph */
    FILE * fout = open_f(TMP_FILE, "w");
    graph_write_file(graph, fout);
    fclose(fout);

    FILE * fin = open_f(TMP_FILE, "r");
    FILE * gold = open_f(graphs[i], "r");

    /* check file lengths lengths */
    fseek(fin , 0 , SEEK_END);
    fseek(gold , 0 , SEEK_END);
    long length_fin  = ftell(fin);
    long length_gold = ftell(gold);
    ASSERT_EQUAL(length_gold, length_fin);
    rewind(fin);
    rewind(gold);

    /* compare each byte */
    char fbyte;
    char gbyte;
    for(long byte=0; byte < length_fin; ++byte) {
      fread(&fbyte, 1, 1, fin);
      fread(&gbyte, 1, 1, gold);
      ASSERT_EQUAL(gbyte, fbyte);
    }

    /* clean up */
    fclose(gold);
    fclose(fin);
    unlink(TMP_FILE);
    graph_free(graph);
  }
}
