
#include "../src/io.h"

#include "ctest/ctest.h"

#include "splatt_test.h"

#include <unistd.h>

static char const * const TMP_FILE = "tmp.txt";


CTEST_DATA(graph)
{
  idx_t ntensors;
  splatt_coo * tensors[MAX_DSETS];
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
    splatt_coo * const tt = data->tensors[i];

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
    remove(TMP_FILE);
    graph_free(graph);
  }
}


#ifdef SPLATT_USE_METIS
CTEST2(graph, metis_part)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    splatt_coo * const tt = data->tensors[i];
    splatt_graph * graph = graph_convert(tt);

    idx_t mycut = 0;
    idx_t cut = 0;
    idx_t * parts = metis_part(graph, 8, &cut);

    /* now make sure cut actually matches */
    for(idx_t v=0; v < graph->nvtxs; ++v) {
      idx_t partA = parts[v];
      for(idx_t e=graph->eptr[v]; e < graph->eptr[v+1]; ++e) {
        idx_t partB = parts[graph->eind[e]];

        if(partA != partB) {
          if(graph->ewgts != NULL) {
            mycut += graph->ewgts[e];
          } else {
            ++mycut;
          }
        }
      }
    }
    /* / 2 because of undirected graph */
    ASSERT_EQUAL(cut, mycut / 2);

    splatt_free(parts);
    graph_free(graph);
  }
}
#endif
