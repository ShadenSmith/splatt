
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../completion/completion.h"
#include "../stats.h"
#include <math.h>
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
#define TC_NORAND_PER_ITERATION 249
#define TC_CSF 248
#define TC_NSTRATUM 247
static struct argp_option tc_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum iterations/epochs (default: 500)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
  {"alg", 'a', "ALG", 0, "which opt algorithm to use (default: sgd)"},
  {"nowrite", TC_NOWRITE, 0, 0, "do not write output to file"},
  {"step", 's', "SIZE", 0, "step size (learning rate) for SGD (default 0.001)"},
  {"reg", TC_REG, "SIZE", 0, "regularization parameter (default CCD/ALS 0.2 SGD 0.005 GD/NLCG/LBFGS 0.01)"},
  {"inner", TC_INNER, "NITERS", 0, "number of inner iterations to use during CCD++ (default: 1)"},
  {"seed", TC_SEED, "SEED", 0, "random seed (default: system time)"},
  {"time", TC_TIME, "SECONDS", 0, "maximum number of seconds, <= 0 to disable (default: 1000)"},
  {"tol", TC_TOL, "TOLERANCE", 0, "converge if RMSE-vl has not improved by TOLERANCE in 20 epochs (default: 1e-4)"},
  {"norand", TC_NORAND_PER_ITERATION, 0, 0, "do not randomly permute every iteration for SGD"},
  {"csf", TC_CSF, 0, 0, "CSF for SGD (default no)"},
  {"stratum", TC_NSTRATUM, "NSTRATUM", 0, "# of stratums for SGD (default P^(M-1) where P is # of ranks and M is # of modes)"},
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

  bool rand_per_iteration;
  bool csf;
  idx_t nstratum;
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
  args->rand_per_iteration = true;
  args->csf = false;
  args->nstratum = -1;
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
  case TC_NORAND_PER_ITERATION:
    args->rand_per_iteration = false;
    break;
  case TC_CSF:
    args->csf = true;
    break;
  case TC_NSTRATUM:
    args->nstratum = atoi(arg);
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

/* defined in mpi_io.c */
void p_get_best_mpi_dim(
  rank_info * const rinfo);

void p_find_layer_boundaries(
  idx_t ** const ssizes,
  idx_t const mode,
  rank_info * const rinfo);

void p_fill_ssizes(
  sptensor_t const * const tt,
  idx_t ** const ssizes,
  rank_info const * const rinfo);

void p_setup_3d(
  rank_info * const rinfo);

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
#ifdef SPLATT_USE_MPI
  rank_info rinfo;
  MPI_Comm_rank(MPI_COMM_WORLD, &rinfo.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rinfo.npes);

  if(rinfo.rank == 0) {
    print_header();
  }
#else
  print_header();
#endif

  srand(args.seed + rinfo.rank);

#ifdef SPLATT_USE_MPI
  if(0 == rinfo.rank) {
    // only support binary file
    FILE *fin = fopen(args.ifnames[0], "r");
    if(fin == NULL) {
      fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n", args.ifnames[0]);
      return -1;
    }
    bin_header header;
    read_binary_header(fin, &header);
    fill_binary_idx(&rinfo.nmodes, 1, &header, fin);
    fill_binary_idx(rinfo.global_dims, rinfo.nmodes, &header, fin);
    fclose(fin);
  }
  MPI_Bcast(&rinfo.nmodes, 1, SPLATT_MPI_IDX, 0, MPI_COMM_WORLD);
  MPI_Bcast(rinfo.global_dims, rinfo.nmodes, SPLATT_MPI_IDX, 0, MPI_COMM_WORLD);
  int nmodes = rinfo.nmodes;

  /* Find p*p*p decomposition */
  rank_info fine_rinfo;
  memcpy(&fine_rinfo, &rinfo, sizeof(rank_info));
  idx_t * ssizes[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    ssizes[m] = (idx_t *)calloc(fine_rinfo.global_dims[m], sizeof(idx_t));
    fine_rinfo.layer_starts[m] = 0;
    fine_rinfo.coords_3d[m] = rinfo.rank;
    fine_rinfo.dims_3d[m] = rinfo.npes;
  }
  fine_rinfo.comm_3d = MPI_COMM_WORLD;
  sptensor_t * train_buf = mpi_simple_distribute(args.ifnames[0], MPI_COMM_WORLD);
  MPI_Allreduce(&train_buf->nnz, &fine_rinfo.global_nnz, 1, SPLATT_MPI_IDX, MPI_SUM, MPI_COMM_WORLD);
  //printf("[%d] train_buf->nnz=%ld train_buf->global_nnz=%ld\n", rinfo.rank, train_buf->nnz, fine_rinfo.global_nnz);
  p_fill_ssizes(train_buf, ssizes, &fine_rinfo);
  for(idx_t m=0; m < nmodes; ++m) {
    p_find_layer_boundaries(ssizes, m, &fine_rinfo);
    free(ssizes[m]);
  }

  /* Find stratum decomposition */
  int min_global_dim = INT_MAX;
  for(int m=1; m < nmodes; ++m) {
    min_global_dim = SS_MIN(min_global_dim, rinfo.global_dims[m]);
  }

  int nstratum_layer = -1;
  if(args.nstratum == -1) {
    nstratum_layer = SS_MIN(rinfo.npes, min_global_dim);
  }
  else {
    nstratum_layer = SS_MIN(SS_MIN(floor(pow(args.nstratum, 1./(nmodes - 1))), min_global_dim), rinfo.npes);
  }
  args.nstratum = 1;
  for(int m=1; m < nmodes; ++m) {
    args.nstratum *= nstratum_layer;
  }
  if(rinfo.rank == 0) {
    printf("Using %d^%d=%ld stratums\n", nstratum_layer, nmodes - 1, args.nstratum);
  }

  int p_per_stratum = (rinfo.npes + nstratum_layer - 1)/nstratum_layer;
  rank_info per_stratum_layer_rinfo;
  memcpy(&per_stratum_layer_rinfo, &rinfo, sizeof(rank_info));
#define SGD_1D
#ifdef SGD_1D
  if(0 == rinfo.rank) printf("%s:%d\n", __FILE__, __LINE__);
  per_stratum_layer_rinfo.dims_3d[0] = p_per_stratum;
  for(int m=1; m < nmodes; ++m) {
    per_stratum_layer_rinfo.dims_3d[m] = 1;
  }
#else
  per_stratum_layer_rinfo.npes = p_per_stratum;
  /*per_stratum_layer_rinfo.global_dims[0] = 1;
  for(int m=1; m < nmodes; ++m) {
    per_stratum_layer_rinfo.global_dims[m] = nstratum_layer;
  }*/
  per_stratum_layer_rinfo.global_dims[0] /= nstratum_layer;
  p_get_best_mpi_dim(&per_stratum_layer_rinfo);
#endif

  if(0 == rinfo.rank) {
    printf("processor decomposition of each stratum layer is %d", per_stratum_layer_rinfo.dims_3d[0]);
    for(int m=1; m < nmodes; ++m) {
      printf("x%d", per_stratum_layer_rinfo.dims_3d[m]);
    }
    printf("\n");
  }

  /* get processor decomposition */
  rinfo.dims_3d[0] = SS_MIN(per_stratum_layer_rinfo.dims_3d[0]*nstratum_layer, rinfo.npes);
  rinfo.layer_ptrs[0] = (idx_t *)splatt_malloc(sizeof(idx_t)*(rinfo.dims_3d[0] + 1));
  for(int l=0; l <= rinfo.dims_3d[0]; ++l) {
    int nsharers = p_per_stratum/per_stratum_layer_rinfo.dims_3d[0];
    rinfo.layer_ptrs[0][l] = fine_rinfo.layer_ptrs[0][SS_MIN(nsharers*l, rinfo.npes)];
    //printf("[%d] rinfo.layer_ptrs[%d][%d] = %ld %d\n", rinfo.rank, 0, l, rinfo.layer_ptrs[0][l], SS_MIN(nsharers*l, rinfo.npes));
  }
  for(int m=1; m < nmodes; ++m) {
    rinfo.dims_3d[m] = per_stratum_layer_rinfo.dims_3d[m];
    rinfo.layer_ptrs[m] = (idx_t *)splatt_malloc(sizeof(idx_t)*(rinfo.dims_3d[m] + 1));
    int nsharers = p_per_stratum/per_stratum_layer_rinfo.dims_3d[m];
    for(int l=0; l <= rinfo.dims_3d[m]; ++l) {
      rinfo.layer_ptrs[m][l] = fine_rinfo.layer_ptrs[m][SS_MIN(nstratum_layer*nsharers*l, rinfo.npes)];
      //printf("[%d] rinfo.layer_ptrs[%d][%d] = %ld %d\n", rinfo.rank, m, l, rinfo.layer_ptrs[m][l], SS_MIN(nstratum_layer*nsharers*l, rinfo.npes));
    }
  }
  p_setup_3d(&rinfo);
  for(int m=0; m < nmodes; ++m) {
    rinfo.layer_starts[m] = rinfo.layer_ptrs[m][rinfo.coords_3d[m]];
    rinfo.layer_ends[m] = rinfo.layer_ptrs[m][rinfo.coords_3d[m] + 1];
    //printf("[%d] m=%d %ld-%ld\n", rinfo.rank, m, rinfo.layer_starts[m], rinfo.layer_ends[m]);
  }
  if(0 == rinfo.rank) {
    printf("total processor decomposition is %d", rinfo.dims_3d[0]);
    for(int m=1; m < nmodes; ++m) {
      printf("x%d", rinfo.dims_3d[m]);
    }
    printf("\n");
  }

  /* load matrix following processor decomposition */
  double t = omp_get_wtime();
  rinfo.global_nnz = fine_rinfo.global_nnz;
  sptensor_t * train = mpi_rearrange_by_rinfo(
    train_buf, &rinfo, MPI_COMM_WORLD);
#pragma omp parallel for
  for(idx_t n=0; n < train->nnz; ++n) {
    for(int m=0; m < nmodes; ++m) {
      assert(train->ind[m][n] >= rinfo.layer_starts[m]);
      assert(train->ind[m][n] < rinfo.layer_ends[m]);
      train->ind[m][n] -= rinfo.layer_starts[m];
    }
  }
  for(int m=0; m < nmodes; ++m) {
    train->dims[m] = rinfo.layer_ends[m] - rinfo.layer_starts[m];
  }
  //printf("[%d] train->nnz=%ld train->nnz=%ld\n", rinfo.rank, train->nnz, rinfo.global_nnz);
  tt_free(train_buf);
  //if(rinfo.rank == 0) printf("%s:%d %g\n", __FILE__, __LINE__, omp_get_wtime() - t);
  t = omp_get_wtime();

  /* decompose validate tensor with the same partitioning used for train */
  timer_start(&timers[TIMER_IO]);
  sptensor_t * validate_buf = mpi_simple_distribute(args.ifnames[1], MPI_COMM_WORLD);
  sptensor_t * validate = mpi_rearrange_by_rinfo(validate_buf, &rinfo, MPI_COMM_WORLD);
  tt_free(validate_buf);
  //if(rinfo.rank == 0) printf("%s:%d %g\n", __FILE__, __LINE__, omp_get_wtime() - t);

#pragma omp parallel for
  for(idx_t n=0; n < validate->nnz; ++n) {
    for(int m=0; m < nmodes; ++m) {
      assert(validate->ind[m][n] >= rinfo.layer_starts[m]);
      assert(validate->ind[m][n] < rinfo.layer_ends[m]);
      validate->ind[m][n] -= rinfo.layer_starts[m];
    }
  }
  for(int m=0; m < nmodes; ++m) {
    validate->dims[m] = rinfo.layer_ends[m] - rinfo.layer_starts[m];
  }

  idx_t global_validate_nnz = validate->nnz;
  MPI_Allreduce(MPI_IN_PLACE, &global_validate_nnz, 1, SPLATT_MPI_IDX, MPI_SUM, MPI_COMM_WORLD);
  timer_stop(&timers[TIMER_IO]);

#else
  sptensor_t * train = tt_read(args.ifnames[0]);
  sptensor_t * validate = tt_read(args.ifnames[1]);
#endif

  if(train == NULL || validate == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  /* print basic tensor stats */
#ifdef SPLATT_USE_MPI
  if(rinfo.rank == 0) {
    mpi_global_stats(train, &rinfo, args.ifnames[0]);
  }
#else
  stats_tt(train, args.ifnames[0], STATS_BASIC, 0, NULL);
#endif

  /* allocate model + workspace */
  tc_model * model = tc_model_alloc(train, args.nfactors, args.which_alg);
  /* scale initial model values by 1/sqrt(nfactors) */
  /*val_t scale = 1/sqrt(model->rank);
  for(int m=0; m < model->nmodes; ++m) {
    for(idx_t i=0; i < model->dims[m]*model->rank; ++i) {
      model->factors[m][i] *= scale;
    }
  }*/
  omp_set_num_threads(args.nthreads);
  tc_ws * ws = tc_ws_alloc(train, model, args.nthreads);

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
  ws->rand_per_iteration = args.rand_per_iteration;
  ws->csf = args.csf;
  ws->nstratum = args.nstratum;

#ifdef SPLATT_USE_MPI
  if(rinfo.rank==0) {
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
  printf("RANKS=%d THREADS=%"SPLATT_PF_IDX"\nSTEP=%0.3e REG=%0.3e\n",
       rinfo.npes, ws->nthreads, ws->learn_rate, ws->regularization[0]);
#else
  printf("THREADS=%"SPLATT_PF_IDX"\nSTEP=%0.3e REG=%0.3e\n",
       ws->nthreads, ws->learn_rate, ws->regularization[0]);
#endif
  printf("VALIDATION=%s\n", args.ifnames[1]);
  if(args.ifnames[2] != NULL) {
    printf("TEST=%s\n", args.ifnames[2]);
  }
#ifdef SPLATT_USE_MPI
  }
#endif

  switch(args.which_alg) {
  case SPLATT_TC_GD:
    printf("ALG=GD\n\n");
    splatt_tc_gd(train, validate, model, ws);
    break;
  case SPLATT_TC_NLCG:
    printf("ALG=NLCG\n\n");
    splatt_tc_nlcg(train, validate, model, ws);
    break;
  case SPLATT_TC_LBFGS:
    printf("ALG=LBFGS\n\n");
    splatt_tc_lbfgs(train, validate, model, ws);
    break;
  case SPLATT_TC_SGD:
#ifdef SPLATT_USE_MPI
    if(rinfo.rank==0) {
      printf("ALG=SGD rand_per_iteration=%d nstratum=%d csf=%d\n\n", ws->rand_per_iteration, ws->nstratum, ws->csf);
    }
    for(int m=0; m < nmodes; ++m) {
      rinfo.mat_ptrs[m] = (idx_t *)splatt_malloc(sizeof(idx_t)*(rinfo.layer_size[m] + 1));
      for(int p=0; p <= rinfo.layer_size[m]; ++p) {
        rinfo.mat_ptrs[m][p] =
          fine_rinfo.layer_ptrs[m][rinfo.layer_size[m]*rinfo.coords_3d[m] + p] -
          fine_rinfo.layer_ptrs[m][rinfo.layer_size[m]*rinfo.coords_3d[m]];
      }
    }
    ws->rinfo = &rinfo;
    ws->global_validate_nnz = global_validate_nnz;
    splatt_tc_sgd(train, validate, model, ws);
#else
    printf("ALG=SGD rand_per_iteration=%d csf=%d\n\n", ws->rand_per_iteration, ws->csf);
    splatt_tc_sgd(train, validate, model, ws);
#endif
    break;
  case SPLATT_TC_CCD:
    printf("ALG=CCD\n\n");
    splatt_tc_ccd(train, validate, model, ws);
    break;
  case SPLATT_TC_ALS:
    printf("ALG=ALS\n\n");
    splatt_tc_als(train, validate, model, ws);
    break;
  default:
    /* error */
    fprintf(stderr, "\n\nSPLATT: unknown completion algorithm\n");
    return SPLATT_ERROR_BADINPUT;
  }

#ifdef SPLATT_USE_MPI
  double mae = tc_mae(validate, ws->best_model, ws);
  if(rinfo.rank == 0) {
    //printf("\nvalidation nnz: %"SPLATT_PF_IDX"\n", ws->global_validate_nnz);
    //printf("BEST VALIDATION RMSE: %0.5f MAE: %0.5f (epoch %"SPLATT_PF_IDX")\n\n",
        //ws->best_rmse, mae, ws->best_epoch);
  }
  //splatt_free(rinfo.send_reqs);
  //splatt_free(rinfo.recv_reqs);
  //splatt_free(rinfo.stats);
  for(idx_t m=0; m < nmodes; ++m) {
    splatt_free(fine_rinfo.layer_ptrs[m]);
  }
#else
  printf("\nvalidation nnz: %"SPLATT_PF_IDX"\n", validate->nnz);
  printf("BEST VALIDATION RMSE: %0.5f MAE: %0.5f (epoch %"SPLATT_PF_IDX")\n\n",
      ws->best_rmse, tc_mae(validate, ws->best_model, ws), ws->best_epoch);
#endif

  tt_free(validate);
  tt_free(train);
  //tc_model_free(model);

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


