#include "../src/csf.h"
#include "../src/cpd/cpd.h"
#include "../src/sptensor.h"

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
  ASSERT_EQUAL(true, opts->unconstrained);

  ASSERT_NOT_NULL(opts);

  splatt_free_cpd_opts(opts);
}


CTEST(cpd, cpd_add_constraints)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  for(idx_t m=0; m < SPLATT_MAX_NMODES; ++m) {
    ASSERT_EQUAL(SPLATT_CON_NONE, opts->constraints[m].which);
    ASSERT_EQUAL(0, opts->chunk_sizes[m]);
    ASSERT_NULL(opts->constraints[m].data);
  }

  /* just one mode */
  splatt_cpd_con_nonneg(opts, 0);
  ASSERT_EQUAL(SPLATT_CON_NONNEG, opts->constraints[0].which);
  ASSERT_NULL(opts->constraints[0].data);
  ASSERT_EQUAL(false, opts->unconstrained);

  /* clear */
  splatt_cpd_con_clear(opts, 0);
  ASSERT_EQUAL(SPLATT_CON_NONE, opts->constraints[0].which);
  ASSERT_NULL(opts->constraints[0].data);
  ASSERT_EQUAL(true, opts->unconstrained);

  splatt_free_cpd_opts(opts);
}

CTEST(cpd, cpd_add_constraints_allmodes)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  splatt_cpd_con_nonneg(opts, MAX_NMODES);
  ASSERT_EQUAL(false, opts->unconstrained);

  for(idx_t m=0; m < MAX_NMODES; ++m) {
    ASSERT_EQUAL(SPLATT_CON_NONNEG, opts->constraints[m].which);
    ASSERT_NULL(opts->constraints[m].data);
  }

  splatt_free_cpd_opts(opts);
}



CTEST(cpd, cpd_add_constraints_clear)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();
  ASSERT_EQUAL(true, opts->unconstrained);

  splatt_cpd_reg_l1(opts, MAX_NMODES, 0.01);
  ASSERT_EQUAL(false, opts->unconstrained);

  splatt_cpd_con_clear(opts, MAX_NMODES);
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    ASSERT_EQUAL(SPLATT_CON_NONE, opts->constraints[m].which);
    ASSERT_NULL(opts->constraints[m].data);
  }

  ASSERT_EQUAL(true, opts->unconstrained);

  splatt_free_cpd_opts(opts);
}


CTEST(cpd, cpd_add_constraints_param)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  val_t reg = 0.01;

  /* L1 */
  splatt_cpd_reg_l1(opts, 0, reg);
  ASSERT_NOT_NULL(opts->constraints[0].data);
  ASSERT_DBL_NEAR_TOL(reg, *((val_t *) opts->constraints[0].data), 0.);

  splatt_cpd_con_clear(opts, 0);

  /* L2 */
  splatt_cpd_reg_l2(opts, 0, reg);
  ASSERT_NOT_NULL(opts->constraints[0].data);
  ASSERT_DBL_NEAR_TOL(reg, *((val_t *) opts->constraints[0].data), 0.);

  splatt_free_cpd_opts(opts);
}


CTEST(cpd, cpd_add_constraints_overwrite)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();

  val_t reg = 0.01;

  splatt_cpd_reg_l1(opts, MAX_NMODES, reg);
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    ASSERT_EQUAL(SPLATT_REG_L1, opts->constraints[m].which);
    ASSERT_NOT_NULL(opts->constraints[m].data);
    ASSERT_DBL_NEAR_TOL(reg, *((val_t *) opts->constraints[m].data), 0.);
  }

  splatt_cpd_con_nonneg(opts, 1);
  ASSERT_EQUAL(SPLATT_CON_NONNEG, opts->constraints[1].which);
  ASSERT_NULL(opts->constraints[1].data);
  /* mode 0 still intact? */
  ASSERT_NOT_NULL(opts->constraints[0].data);
  ASSERT_DBL_NEAR_TOL(reg, *((val_t *) opts->constraints[0].data), 0.);

  splatt_free_cpd_opts(opts);
}


CTEST(cpd, cpd)
{

}
