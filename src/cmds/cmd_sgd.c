
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../kruskal.h"
#include "../sgd.h"
#include "../stats.h"


/******************************************************************************
 * SPLATT-SGD
 *****************************************************************************/
int splatt_sgd_cmd(
  int argc,
  char ** argv)
{
  if(argc == 1) {
    fprintf(stderr, "splatt-sgd needs tensor\n");
    return SPLATT_ERROR_BADINPUT;
  }

  sptensor_t * train = tt_read(argv[1]);
  if(train == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  /* print basic tensor stats? */
  stats_tt(train, argv[1], STATS_BASIC, 0, NULL);

  idx_t const nfactors = 50;

  splatt_kruskal model;
  model.rank = nfactors;
  model.nmodes = train->nmodes;
  model.lambda = splatt_malloc(nfactors * sizeof(*model.lambda));

  /* allocate */
  for(idx_t m=0; m < train->nmodes; ++m) {
    matrix_t * tmp = mat_rand(train->dims[m], nfactors);
    model.dims[m] = train->dims[m];
    model.factors[m] = tmp->vals;

    /* clean up */
    free(tmp);
  }

  splatt_sgd(train, &model);

  tt_free(train);

  /* test rmse */
  sptensor_t * test = tt_read(argv[2]);
  stats_tt(test, argv[1], STATS_BASIC, 0, NULL);


  printf("TEST RMSE: %0.5f\n", kruskal_rmse(test, &model));

  tt_free(test);
  splatt_free_kruskal(&model);

  return EXIT_SUCCESS;
}


