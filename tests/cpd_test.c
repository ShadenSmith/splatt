#include "../src/csf.h"
#include "../src/cpd/cpd.h"
#include "../src/sptensor.h"
#include "../src/util.h"

#include "ctest/ctest.h"
#include "splatt_test.h"


CTEST_DATA(cpd)
{
  sptensor_t * tt;
  double * opts;
};

CTEST_SETUP(cpd)
{
  data->tt = tt_read(DATASET(med4.tns));

  /* setup opts */
  data->opts = splatt_default_opts();
  data->opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  data->opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  data->opts[SPLATT_OPTION_NTHREADS] = 2;
}

CTEST_TEARDOWN(cpd)
{
  tt_free(data->tt);
  free(data->opts);
}



CTEST(cpd, cpd_opt_alloc)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  ASSERT_NOT_NULL(opts);
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    ASSERT_NOT_NULL(opts->constraints[m]);
    ASSERT_EQUAL(0, strcmp(opts->constraints[m]->description, "UNCONSTRAINED"));
  }

  splatt_free_cpd_opts(opts);
}

CTEST2(cpd, cpd_alloc_ws)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();
  splatt_global_opts * gopts = splatt_alloc_global_opts();

  double * csf_opts = splatt_default_opts();
  csf_opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  splatt_csf * csf = csf_alloc(data->tt, csf_opts);

  /* Allocate an unconstrained ws */

  cpd_ws * ws = cpd_alloc_ws(csf, 10, opts, gopts);

  ASSERT_TRUE(ws->unconstrained);
  for(idx_t m=0; m < csf->nmodes; ++m) {
    ASSERT_NULL(ws->duals[m]);
  }
  ASSERT_NULL(ws->auxil);
  ASSERT_NULL(ws->mat_init);

  cpd_free_ws(ws);

  /* Register a constraint */
  idx_t mode = 0;
  splatt_register_nonneg(opts, &mode, 1);

  ws = cpd_alloc_ws(csf, 10, opts, gopts);
  ASSERT_FALSE(ws->unconstrained);
  ASSERT_NOT_NULL(ws->duals[0]);
  for(idx_t m=1; m < csf->nmodes; ++m) {
    ASSERT_NULL(ws->duals[m]);
  }
  ASSERT_NOT_NULL(ws->auxil);
  ASSERT_NOT_NULL(ws->mat_init);
  cpd_free_ws(ws);

  csf_free(csf, csf_opts);
  splatt_free_opts(csf_opts);
  splatt_free_global_opts(gopts);
  splatt_free_cpd_opts(opts);
}


CTEST(cpd, cpd_register_constraints)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  for(idx_t m=0; m < SPLATT_MAX_NMODES; ++m) {
    ASSERT_NOT_NULL(opts->constraints[m]);
    ASSERT_NULL(opts->constraints[m]->data);
  }

  splatt_cpd_constraint * con = splatt_alloc_constraint(SPLATT_CON_CLOSEDFORM);
  sprintf(con->description, "TEST");

  /* initial register */
  splatt_register_constraint(opts, 0, con);
  ASSERT_EQUAL(0, strcmp("TEST", opts->constraints[0]->description));

  /* overwrite */
  con = splatt_alloc_constraint(SPLATT_CON_CLOSEDFORM);
  sprintf(con->description, "TEST2");
  splatt_register_constraint(opts, 0, con);
  ASSERT_EQUAL(0, strcmp("TEST2", opts->constraints[0]->description));

  splatt_free_cpd_opts(opts);
}



/*
 * NON-NEGATIVE
 */
void splatt_nonneg_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize);
CTEST(cpd, cpd_constraint_nonneg)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  idx_t mode = 1;
  int success = splatt_register_nonneg(opts, &mode, 1);
  ASSERT_EQUAL(SPLATT_SUCCESS, success);
  ASSERT_EQUAL(0, strcmp(opts->constraints[mode]->description, "NON-NEGATIVE"));

  /* test the projection */
  idx_t nrows = 10;
  idx_t ncols = 2;
  val_t * vals = splatt_malloc(nrows * ncols * sizeof(*vals));

  fill_rand(vals, nrows * ncols);
  vals[0] = -1.0; /* force at least one negative */

  splatt_nonneg_prox(vals, nrows, ncols, 0, NULL, 0, true);

  for(idx_t x=0; x < nrows * ncols; ++x) {
    ASSERT_TRUE(vals[x] >= 0.);
  }

  splatt_free(vals);
  splatt_free_cpd_opts(opts);
}


/*
 * LASSO
 */
void splatt_lasso_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize);
CTEST(cpd, cpd_constraint_lasso)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  idx_t mode = 1;
  val_t const mult = 0.01;
  int success = splatt_register_lasso(opts, mult, &mode, 1);
  ASSERT_EQUAL(SPLATT_SUCCESS, success);

  /* test the projection */
  idx_t nrows = 10;
  idx_t ncols = 2;
  val_t * vals = splatt_malloc(nrows * ncols * sizeof(*vals));
  fill_rand(vals, nrows * ncols);

  vals[0] = mult; /* force a 0 */
  vals[1] =  1. + mult;
  vals[2] = -1. - mult;

  splatt_lasso_prox(vals, nrows, ncols, 0, (void *) &mult, 1., true);

  ASSERT_DBL_NEAR_TOL( 0., vals[0], 0.);
  ASSERT_DBL_NEAR_TOL( 1., vals[1], 0.);
  ASSERT_DBL_NEAR_TOL(-1., vals[2], 0.);

  splatt_free(vals);
  splatt_free_cpd_opts(opts);
}


/*
 * ROW SIMPLEX
 */
void splatt_rowsimp_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize);
CTEST(cpd, cpd_constraint_rowsimp)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  idx_t mode = 1;
  int success = splatt_register_rowsimp(opts, &mode, 1);
  ASSERT_EQUAL(SPLATT_SUCCESS, success);

  /* test the projection */
  idx_t nrows = 100;
  idx_t ncols = 11;
  val_t * vals = splatt_malloc(nrows * ncols * sizeof(*vals));
  fill_rand(vals, nrows * ncols);

  splatt_rowsimp_prox(vals, nrows, ncols, 0, NULL, 1., true);

  for(idx_t i=0; i < nrows; ++i) {
    val_t sum = 0;
    for(idx_t j=0; j < ncols; ++j) {
      sum += vals[j + (i*ncols)];

      ASSERT_TRUE(vals[j + (i*ncols)] >= 0.);
    }
    ASSERT_DBL_NEAR_TOL(1.0, sum, 1e-6);
  }

  splatt_free(vals);
  splatt_free_cpd_opts(opts);
}


void splatt_colsimp_prox(
    val_t * primal,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const offset,
    void * data,
    val_t const rho,
    bool const should_parallelize);
CTEST(cpd, cpd_constraint_colsimp)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  idx_t mode = 1;
  int success = splatt_register_colsimp(opts, &mode, 1);
  ASSERT_EQUAL(SPLATT_SUCCESS, success);

  /* test the projection */
  idx_t nrows = 100;
  idx_t ncols = 11;
  val_t * vals = splatt_malloc(nrows * ncols * sizeof(*vals));
  fill_rand(vals, nrows * ncols);

  splatt_colsimp_prox(vals, nrows, ncols, 0, NULL, 1., true);

  for(idx_t j=0; j < ncols; ++j) {
    val_t sum = 0;
    for(idx_t i=0; i < nrows; ++i) {
      sum += vals[j + (i*ncols)];

      ASSERT_TRUE(vals[j + (i*ncols)] >= 0.);
    }
    ASSERT_DBL_NEAR_TOL(1.0, sum, 1e-6);
  }

  splatt_free(vals);
  splatt_free_cpd_opts(opts);
}

