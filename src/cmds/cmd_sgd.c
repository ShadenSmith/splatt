
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../completion.h"
#include "../sgd.h"
#include "../stats.h"

#include <omp.h>


/******************************************************************************
 * SPLATT-SGD
 *****************************************************************************/
int splatt_sgd_cmd(
  int argc,
  char ** argv)
{
  if(argc < 4) {
    fprintf(stderr, "splatt-sgd <train> <validate> <test>\n");
    return SPLATT_ERROR_BADINPUT;
  }

  sptensor_t * train = tt_read(argv[1]);
  if(train == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  /* print basic tensor stats? */
  stats_tt(train, argv[1], STATS_BASIC, 0, NULL);

  idx_t const nmodes = train->nmodes;
  idx_t const nfactors = 10;

  tc_model * model = tc_model_alloc(train, nfactors, SPLATT_TC_SGD);

  sptensor_t * validate = tt_read(argv[2]);
  if(validate == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }
  printf("validate nnz: %"SPLATT_PF_IDX"\n\n", validate->nnz);

  idx_t nthreads = omp_get_max_threads();
  tc_ws * ws = tc_ws_alloc(model, nthreads);

  printf("lrn: %0.3e  reg: %0.3e\n\n", ws->learn_rate, ws->regularization[0]);

  splatt_sgd(train, validate, model, ws);

  tt_free(validate);
  tt_free(train);

  /* test rmse */
  sptensor_t * test = tt_read(argv[3]);
  printf("test nnz: %"SPLATT_PF_IDX"\n", test->nnz);
  printf("TEST RMSE: %0.5f\n", tc_rmse(test, model, ws));

  /* write output */
  for(idx_t m=0; m < nmodes; ++m) {
    char * matfname = NULL;
    asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);

    matrix_t tmpmat;
    tmpmat.rowmajor = 1;
    tmpmat.I = model->dims[m];
    tmpmat.J = nfactors;
    tmpmat.vals = model->factors[m];

    mat_write(&tmpmat, matfname);
    free(matfname);
  }

  tt_free(test);
  tc_model_free(model);
  tc_ws_free(ws);

  return EXIT_SUCCESS;
}


