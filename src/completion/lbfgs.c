#include <math.h>

#include "completion.h"
#include "gradient.h"
#include "../csf.h"
#include "../util.h"
#include "liblbfgs/lbfgs.h"
#include "liblbfgs/arithmetic_ansi.h"

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

typedef struct
{
  sptensor_t *train;
  const sptensor_t *validate;
  splatt_csf *csf;
  tc_model *model;
  tc_ws *ws;
  val_t * gradients[MAX_NMODES];
} lbfgs_user_data;


static lbfgsfloatval_t p_lbfgs_evaluate(
    void *instance, /* the user data sent for lbfgs() function by the client */
    const lbfgsfloatval_t *x, /* the current values of variables */
    lbfgsfloatval_t *g, /* gradient vector */
    const int n, /* number of variables */
    const lbfgsfloatval_t step) /* current step of the line search routine */
{
  lbfgs_user_data *user_data = (lbfgs_user_data *)instance;

  sptensor_t *train = user_data->train;
  splatt_csf *csf = user_data->csf;
  tc_model *model = user_data->model;
  tc_ws *ws = user_data->ws;
  val_t * * gradients = user_data->gradients;

  idx_t const nmodes = model->nmodes;

  timer_start(&ws->train_time);
  tc_gradient(csf, model, ws, gradients);

  idx_t offset = 0;
  for(idx_t m=0; m < nmodes; ++m) {
    size_t const bytes = model->dims[m] * model->rank * sizeof(*g);
    par_memcpy(g + offset, gradients[m], bytes);
    offset += train->dims[m] * model->rank;
  }
  timer_stop(&ws->train_time);

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);


  return loss + frobsq;
}

static int ls_acc = 0;

static int p_lbfgs_progress(
    void *instance, /* the user data sent for lbfgs() function by the client */
    const lbfgsfloatval_t *x, /* the current value of variables */
    const lbfgsfloatval_t *g, /* the current gradient values of variables */
    const lbfgsfloatval_t fx, /* the current value of the objective function */
    const lbfgsfloatval_t xnorm, /* the Euclidean norm of the variables */
    const lbfgsfloatval_t gnorm, /* the Euclidean norm of the gradients */
    const lbfgsfloatval_t step, /* the line-search step used for this iteration */
    int n, /* the number of variables */
    int k, /* the iteration count */
    int ls) /* the number of evaluations called for this iteration */
{
  lbfgs_user_data *user_data = (lbfgs_user_data *)instance;

  sptensor_t *train = user_data->train;
  const sptensor_t *validate = user_data->validate;
  tc_model *model = user_data->model;
  tc_ws *ws = user_data->ws;

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  ls_acc += ls;
  printf("ls: %d\t", ls_acc);
  tc_converge(train, validate, model, loss, frobsq, k, ws);

  return 0;
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void splatt_tc_lbfgs(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws)
{
  /* convert training data to CSF-ALLMODE */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  splatt_csf * csf = csf_alloc(train, opts);
  assert(csf->ntiles == 1);

  idx_t const nmodes = train->nmodes;

  timer_reset(&ws->train_time);
  timer_reset(&ws->test_time);


  idx_t n = 0;
  for(idx_t m=0; m < model->nmodes; ++m) {
    n += model->dims[m] * model->rank;
  }

  val_t *mat = (val_t *)splatt_malloc(sizeof(val_t)*n);
  n = 0;
  for(idx_t m=0; m < model->nmodes; ++m) {
    memcpy(mat + n, model->factors[m], sizeof(val_t)*model->dims[m]*model->rank);
    model->factors[m] = mat + n;
    n += model->dims[m] * model->rank;
  }

  val_t loss = tc_loss_sq(train, model, ws);
  val_t frobsq = tc_frob_sq(model, ws);
  val_t prev_obj = loss + frobsq;
  tc_converge(train, validate, model, loss, frobsq, 0, ws);

  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  param.max_iterations = ws->max_its;
  param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
  //param.min_step = ws->learn_rate;

  lbfgs_user_data user_data;
  user_data.train = train;
  user_data.validate = validate;
  user_data.csf = csf;
  user_data.model = model;
  user_data.ws = ws;

  /* allocate gradients */
  for(idx_t m=0; m < nmodes; ++m) {
    user_data.gradients[m] = splatt_malloc(model->dims[m] * model->rank * sizeof(val_t));
  }


  val_t fx;
  int ret = lbfgs(n, mat, &fx, p_lbfgs_evaluate, p_lbfgs_progress, &user_data, &param);

  if (LBFGS_SUCCESS != ret) {
    printf("L-BFGS finished with status code = %d (", ret);
    switch (ret) {
    case LBFGSERR_UNKNOWNERROR: printf("LBFGSERR_UNKNOWNERROR"); break;
    case LBFGSERR_LOGICERROR: printf("LBFGSERR_LOGICERROR"); break;
    case LBFGSERR_OUTOFMEMORY: printf("LBFGSERR_OUTOFMEMOR"); break;
    case LBFGSERR_CANCELED: printf("LBFGSERR_CANCELED"); break;
    case LBFGSERR_INVALID_N: printf("LBFGSERR_INVALID_N"); break;
    case LBFGSERR_INVALID_N_SSE: printf("LBFGSERR_INVALID_N_SSE"); break;
    case LBFGSERR_INVALID_X_SSE: printf("LBFGSERR_INVALID_X_SSE"); break;
    case LBFGSERR_INVALID_EPSILON: printf("LBFGSERR_INVALID_EPSILON"); break;
    case LBFGSERR_INVALID_TESTPERIOD: printf("LBFGSERR_INVALID_TESTPERIOD"); break;
    case LBFGSERR_INVALID_DELTA: printf("LBFGSERR_INVALID_DELTA"); break;
    case LBFGSERR_INVALID_LINESEARCH: printf("LBFGSERR_INVALID_LINESEARCH"); break;
    case LBFGSERR_INVALID_MINSTEP: printf("LBFGSERR_INVALID_MINSTEP"); break;
    case LBFGSERR_INVALID_MAXSTEP: printf("LBFGSERR_INVALID_MAXSTEP"); break;
    case LBFGSERR_INVALID_FTOL: printf("LBFGSERR_INVALID_FTOL"); break;
    case LBFGSERR_INVALID_WOLFE: printf("LBFGSERR_INVALID_WOLFE"); break;
    case LBFGSERR_INVALID_GTOL: printf("LBFGSERR_INVALID_GTOL"); break;
    case LBFGSERR_INVALID_XTOL: printf("LBFGSERR_INVALID_XTOL"); break;
    case LBFGSERR_INVALID_MAXLINESEARCH: printf("LBFGSERR_INVALID_MAXLINESEARCH"); break;
    case LBFGSERR_INVALID_ORTHANTWISE: printf("LBFGSERR_INVALID_ORTHANTWISE"); break;
    case LBFGSERR_INVALID_ORTHANTWISE_START: printf("LBFGSERR_INVALID_ORTHANTWISE_START"); break;
    case LBFGSERR_INVALID_ORTHANTWISE_END: printf("LBFGSERR_INVALID_ORTHANTWISE_END"); break;
    case LBFGSERR_OUTOFINTERVAL: printf("LBFGSERR_OUTOFINTERVAL"); break;
    case LBFGSERR_INCORRECT_TMINMAX: printf("LBFGSERR_INCORRECT_TMINMAX"); break;
    case LBFGSERR_ROUNDING_ERROR: printf("LBFGSERR_ROUNDING_ERROR"); break;
    case LBFGSERR_MINIMUMSTEP: printf("LBFGSERR_MINIMUMSTEP"); break;
    case LBFGSERR_MAXIMUMSTEP: printf("LBFGSERR_MAXIMUMSTEP"); break;
    case LBFGSERR_MAXIMUMLINESEARCH: printf("LBFGSERR_MAXIMUMLINESEARCH"); break;
    case LBFGSERR_MAXIMUMITERATION: printf("LBFGSERR_MAXIMUMITERATION"); break;
    case LBFGSERR_WIDTHTOOSMALL: printf("LBFGSERR_WIDTHTOOSMALL"); break;
    case LBFGSERR_INVALIDPARAMETERS: printf("LBFGSERR_INVALIDPARAMETERS"); break;
    case LBFGSERR_INCREASEGRADIENT: printf("LBFGSERR_INCREASEGRADIENT"); break;
    default: assert(false);
    }
    printf(")\n");
  }


  /* cleanup */
  for(idx_t m=0; m < nmodes; ++m) {
    splatt_free(user_data.gradients[m]);
  }
  csf_free(csf, opts);
  splatt_free_opts(opts);
}

