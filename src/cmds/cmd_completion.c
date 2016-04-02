
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../completion/completion.h"
#include "../stats.h"
#include "../splatt_mpi.h"

#include <omp.h>




/******************************************************************************
 * ARG PARSING
 *****************************************************************************/

static char tc_args_doc[] = "<train> <validate> [test]";
static char tc_doc[] =
  "splatt-complete -- Complete a tensor with missing entries.\n"
  "Available tensor completion algorithms are:\n"
  "  gd\t\tgradient descent\n"
  "  cg\t\nnonlinear conjugate gradient\n"
  "  lbfgs\t\tlimited-memory BFGS\n"
  "  sgd\t\tstochastic gradient descent\n"
  "  ccd\t\tcoordinate descent\n"
  "  als\t\talternating least squares\n";


#define TC_REG 255
#define TC_NOWRITE 254
#define TC_SEED 253
#define TC_TIME 252
#define TC_TOL 251
#define TC_INNER 250
static struct argp_option tc_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum iterations/epochs (default: 500)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
  {"alg", 'a', "ALG", 0, "which opt algorithm to use (default: sgd)"},
  {"nowrite", TC_NOWRITE, 0, 0, "do not write output to file"},
  {"step", 's', "SIZE", 0, "step size (learning rate) for SGD"},
  {"reg", TC_REG, "SIZE", 0, "step size (learning rate) for SGD"},
  {"inner", TC_INNER, "NITERS", 0, "number of inner iterations to use during CCD++ (default: 1)"},
  {"seed", TC_SEED, "SEED", 0, "random seed (default: system time)"},
  {"time", TC_TIME, "SECONDS", 0, "maximum number of seconds, <= 0 to disable (default: 1000)"},
  {"tol", TC_TOL, "TOLERANCE", 0, "converge if RMSE-vl has not improved by TOLERANCE in 20 epochs (default: 1e-4)"},
  {0}
};


typedef struct
{
  char * name;
  splatt_tc_type which;
} tc_alg_map;

static tc_alg_map maps[] = {
  { "gd", SPLATT_TC_GD },
  { "lbfgs", SPLATT_TC_LBFGS },
  { "cg", SPLATT_TC_NLCG },
  { "nlcg", SPLATT_TC_NLCG },
  { "sgd", SPLATT_TC_SGD },
  { "als", SPLATT_TC_ALS },
  { "ccd", SPLATT_TC_CCD },
  { NULL,  SPLATT_TC_NALGS }
};


/**
* @brief Parse a command into a splatt_tc_type (completion algorithm).
*
* @param arg The string to parse.
*
* @return The optimization algorithm to use. SPLATT_TC_NALGS on error.
*/
static splatt_tc_type parse_tc_alg(
    char const * const arg)
{
  int ptr = 0;
  while(maps[ptr].name != NULL) {
    if(strcmp(arg, maps[ptr].name) == 0) {
      return maps[ptr].which;
    }
    ++ptr;
  }

  /* error */
  return SPLATT_TC_NALGS;
}



typedef struct
{
  /* TRAIN, VALIDATE, TEST */
  char * ifnames[3];
  int nfiles;

  bool set_seed;
  unsigned int seed;

  splatt_tc_type which_alg;

  bool write;

  val_t learn_rate;
  val_t reg;

  bool set_tolerance;
  val_t tolerance;

  idx_t max_its;
  idx_t num_inner;
  bool set_timeout;
  double max_seconds;
  idx_t nfactors;
  idx_t nthreads;
} tc_cmd_args;


/**
* @brief Fill the tensor completion arguments with some sane defaults.
*
* @param args The arguments to initialize.
*/
static void default_tc_opts(
    tc_cmd_args * const args)
{
  args->nfiles = 0;
  for(int n=0; n < 3; ++n) {
    args->ifnames[n] = NULL;
  }

  args->which_alg = SPLATT_TC_SGD;
  args->write = false;
  args->nfactors = 10;
  args->learn_rate = -1.;
  args->reg = -1.;
  args->max_its = 0;
  args->set_timeout = false;
  args->nthreads = omp_get_max_threads();
  args->set_seed = false;
  args->seed = time(NULL);
  args->set_tolerance = false;
  args->num_inner = 1;
}



static error_t parse_tc_opt(
    int key,
    char * arg,
    struct argp_state * state)
{
  tc_cmd_args * args = state->input;
  char * buf;
  int cnt = 0;

  /* -i=50 should also work... */
  if(arg != NULL && arg[0] == '=') {
    ++arg;
  }

  switch(key) {
  case 'i':
    args->max_its = strtoull(arg, &buf, 10);
    break;
  case 't':
    args->nthreads = strtoull(arg, &buf, 10);
    omp_set_num_threads(args->nthreads);
    break;
  case 'v':
    timer_inc_verbose();
    break;
  case 'a':
    args->which_alg = parse_tc_alg(arg);
    if(args->which_alg == SPLATT_TC_NALGS) {
      fprintf(stderr, "SPLATT: unknown completion algorithm '%s'.\n", arg);
      argp_usage(state);
    }
    break;
  case TC_NOWRITE:
    args->write = false;
    break;
  case 'r':
    args->nfactors = strtoull(arg, &buf, 10);
    break;
  case 's':
    args->learn_rate = strtod(arg, &buf);
    break;
  case TC_REG:
    args->reg = strtod(arg, &buf);
    break;
  case TC_SEED:
    args->seed = (unsigned int) atoi(arg);
    args->set_seed = true;
    break;
  case TC_TIME:
    args->max_seconds = atof(arg);
    args->set_timeout = true;
    break;
  case TC_TOL:
    args->tolerance = atof(arg);
    args->set_tolerance = true;
    break;
  case TC_INNER:
    args->num_inner = strtoull(arg, &buf, 10);
    break;

  case ARGP_KEY_ARG:
    if(args->nfiles == 3) {
      argp_usage(state);
      break;
    }
    args->ifnames[args->nfiles++] = arg;
    break;

  case ARGP_KEY_END:
    if(args->ifnames[0] == NULL || args->ifnames[1] == NULL) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}


/* tie it all together */
static struct argp tc_argp = {tc_options, parse_tc_opt, tc_args_doc, tc_doc};


/******************************************************************************
 * SPLATT-COMPLETE
 *****************************************************************************/
int splatt_tc_cmd(
  int argc,
  char ** argv)
{
  tc_cmd_args args;
  default_tc_opts(&args);
  argp_parse(&tc_argp, argc, argv, ARGP_IN_ORDER, 0, &args);
  omp_set_num_threads(args.nthreads);

#ifdef SPLATT_USE_MPI
  /* get global info */
  rank_info rinfo;
  MPI_Comm_rank(MPI_COMM_WORLD, &rinfo.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rinfo.npes);

  if(rinfo.rank == 0) {
    print_header();
  }
#else
  print_header();
#endif

  int success = SPLATT_SUCCESS;

  srand(args.seed);

  /* read input */
#ifdef SPLATT_USE_MPI
  sptensor_t * train = NULL;
  sptensor_t * validate = NULL;

  /* read + distribute tensors */
  switch(args.which_alg) {
  case SPLATT_TC_ALS:
    success = mpi_tc_distribute_coarse(args.ifnames[0], args.ifnames[1],
        NULL, &train, &validate, &rinfo);
    break;
  case SPLATT_TC_CCD:
    success = mpi_tc_distribute_med(args.ifnames[0], args.ifnames[1], NULL,
        &train, &validate, &rinfo);
    break;
  default:
    fprintf(stderr, "SPLATT: alg not supported for distributed execution.\n");
    return EXIT_FAILURE;
  }
  if(success != SPLATT_SUCCESS) {
    return success;
  }

  return EXIT_SUCCESS;


#else
  sptensor_t * train = tt_read(args.ifnames[0]);
  sptensor_t * validate = tt_read(args.ifnames[1]);
#endif
  if(train == NULL || validate == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }
  idx_t const nmodes = train->nmodes;

  /* print basic tensor stats */
#ifdef SPLATT_USE_MPI
  if(rinfo.rank == 0) {
    mpi_global_stats(train, &rinfo, args.ifnames[0]);
  }

  /* determine matrix distribution - this also calls tt_remove_empty() */
  permutation_t * perm = mpi_distribute_mats(&rinfo, train, rinfo.decomp);

  for(idx_t m=0; m < train->nmodes; ++m) {
    /* index into local tensor to grab owned rows */
    mpi_find_owned(train, m, &rinfo);
    /* determine isend and ineed lists */
    mpi_compute_ineed(&rinfo, train, m, args.nfactors, 3);
  }

  mpi_rank_stats(train, &rinfo);

  /* allocate model */
  tc_model * model = mpi_tc_model_alloc(train, args.nfactors, args.which_alg,
      perm, &rinfo);
#else


  stats_tt(train, args.ifnames[0], STATS_BASIC, 0, NULL);
  tc_model * model = tc_model_alloc(train, args.nfactors, args.which_alg);
#endif

  tc_ws * ws = tc_ws_alloc(train, model, args.nthreads);
#ifdef SPLATT_USE_MPI
  ws->rinfo = &rinfo;
#endif

  /* check for non-default vals */
  if(args.learn_rate != -1.) {
    ws->learn_rate = args.learn_rate;
  }
  if(args.reg != -1.) {
    for(idx_t m=0; m < nmodes; ++m) {
      ws->regularization[m] = args.reg;
    }
  }
  if(args.max_its != 0) {
    ws->max_its = args.max_its;
  }
  if(args.set_timeout) {
    ws->max_seconds = args.max_seconds;
  }
  if(args.set_tolerance) {
    ws->tolerance = args.tolerance;
  }
  ws->num_inner = args.num_inner;

#ifdef SPLATT_USE_MPI
  if(rinfo.rank == 0) {
#endif
  printf("Factoring ------------------------------------------------------\n");
  printf("NFACTORS=%"SPLATT_PF_IDX" MAXITS=%"SPLATT_PF_IDX" ",
      model->rank, ws->max_its);
  if(args.set_timeout) {
    printf("MAXTIME=NONE ");
  } else {
    printf("MAXTIME=%0.1fs ", ws->max_seconds);
  }
  printf("TOL=%0.1e ", ws->tolerance);
  if(args.set_seed) {
    printf("SEED=%u ", args.seed);
  } else {
    printf("SEED=time ");
  }
#ifdef SPLATT_USE_MPI
  printf("RANKS=%d ", rinfo.npes);
#endif
  printf("THREADS=%"SPLATT_PF_IDX"\nSTEP=%0.3e REG=%0.3e\n",
       ws->nthreads, ws->learn_rate, ws->regularization[0]);
  printf("VALIDATION=%s\n", args.ifnames[1]);
  if(args.ifnames[2] != NULL) {
    printf("TEST=%s\n", args.ifnames[2]);
  }

  switch(args.which_alg) {
  case SPLATT_TC_GD:
    printf("ALG=GD\n\n");
    break;
  case SPLATT_TC_NLCG:
    printf("ALG=NLCG\n\n");
    break;
  case SPLATT_TC_LBFGS:
    printf("ALG=LBFGS\n\n");
    break;
  case SPLATT_TC_SGD:
    printf("ALG=SGD\n\n");
    break;
  case SPLATT_TC_CCD:
    printf("ALG=CCD\n\n");
    break;
  case SPLATT_TC_ALS:
    printf("ALG=ALS\n\n");
    break;
  default:
    /* error */
    fprintf(stderr, "\n\nSPLATT: unknown completion algorithm\n");
    return SPLATT_ERROR_BADINPUT;
  }

#ifdef SPLATT_USE_MPI
  }
#endif

  switch(args.which_alg) {
  case SPLATT_TC_GD:
    splatt_tc_gd(train, validate, model, ws);
    break;
  case SPLATT_TC_NLCG:
    splatt_tc_nlcg(train, validate, model, ws);
    break;
  case SPLATT_TC_LBFGS:
    splatt_tc_lbfgs(train, validate, model, ws);
    break;
  case SPLATT_TC_SGD:
    splatt_tc_sgd(train, validate, model, ws);
    break;
  case SPLATT_TC_CCD:
    splatt_tc_ccd(train, validate, model, ws);
    break;
  case SPLATT_TC_ALS:
    splatt_tc_als(train, validate, model, ws);
    break;
  default:
    /* error */
    return SPLATT_ERROR_BADINPUT;
  }

#ifdef SPLATT_USE_MPI
  return EXIT_SUCCESS;
#endif

  printf("\nvalidation nnz: %"SPLATT_PF_IDX"\n", validate->nnz);
  printf("BEST VALIDATION RMSE: %0.5f MAE: %0.5f (epoch %"SPLATT_PF_IDX")\n\n",
      ws->best_rmse, tc_mae(validate, ws->best_model, ws), ws->best_epoch);

  tt_free(validate);
  tt_free(train);
  tc_model_free(model);

  /* test rmse on best model found */
  if(args.ifnames[2] != NULL) {
    sptensor_t * test = tt_read(args.ifnames[2]);
    if(test == NULL) {
      return SPLATT_ERROR_BADINPUT;
    }
    printf("test nnz: %"SPLATT_PF_IDX"\n", test->nnz);
    printf("TEST RMSE: %0.5f MAE: %0.5f\n",
        tc_rmse(test, ws->best_model, ws),
        tc_mae(test, ws->best_model, ws));
    tt_free(test);
  }

  /* write the best model */
  if(args.write) {
    for(idx_t m=0; m < nmodes; ++m) {
      char * matfname = NULL;
      asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);

      matrix_t tmpmat;
      tmpmat.rowmajor = 1;
      tmpmat.I = ws->best_model->dims[m];
      tmpmat.J = args.nfactors;
      tmpmat.vals = ws->best_model->factors[m];

      mat_write(&tmpmat, matfname);
      free(matfname);
    }
  }

  tc_ws_free(ws);

  return EXIT_SUCCESS;
}


