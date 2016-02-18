
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
  idx_t const nfactors = 32;

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

  val_t * regs = splatt_malloc(train->nmodes * sizeof(*regs));
  for(idx_t m=0; m < train->nmodes; ++m) {
    regs[m] = 0.02;
  }

  sptensor_t * validate = tt_read(argv[2]);
  if(validate == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }
  printf("validate nnz: %"SPLATT_PF_IDX" (%0.1f%%)\n\n",
      validate->nnz, 100. * (double)validate->nnz / (double)train->nnz);

  splatt_sgd(train, validate, &model, 1000, 0.002, regs);


  splatt_free(regs);

  tt_free(validate);
  tt_free(train);

  /* test rmse */
  sptensor_t * test = tt_read(argv[3]);
  printf("test nnz: %"SPLATT_PF_IDX"\n", test->nnz);
  printf("TEST RMSE: %0.5f\n", kruskal_rmse(test, &model));

  vec_write(model.lambda, nfactors, "lambda.mat");

  /* write output */
  for(idx_t m=0; m < nmodes; ++m) {
    char * matfname = NULL;
    asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);

    matrix_t tmpmat;
    tmpmat.rowmajor = 1;
    tmpmat.I = model.dims[m];
    tmpmat.J = nfactors;
    tmpmat.vals = model.factors[m];

    mat_write(&tmpmat, matfname);
    free(matfname);
  }

  tt_free(test);
  splatt_free_kruskal(&model);

  return EXIT_SUCCESS;
}


