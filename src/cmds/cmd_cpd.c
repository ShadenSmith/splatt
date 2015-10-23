
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../ftensor.h"
#include "../tile.h"
#include "../stats.h"
#include "../cpd.h"
#include "../splatt_mpi.h"
#include "../sort.h"
#include "../util.h"
#include "../timer.h"


/******************************************************************************
 * SPLATT CPD
 *****************************************************************************/
static char cpd_args_doc[] = "TENSOR";
static char cpd_doc[] =
  "splatt-cpd -- Compute the CPD of a sparse tensor.\n";

#define TT_NOWRITE 253
#define TT_TOL 254
#define TT_TILE 255
static struct argp_option cpd_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum number of iterations to use (default: 50)"},
  {"tol", TT_TOL, "TOLERANCE", 0, "minimum change for convergence (default: 1e-5)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"tile", TT_TILE, 0, 0, "use tiling during SPLATT"},
  {"nowrite", TT_NOWRITE, 0, 0, "do not write output to file (default: WRITE)"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
#ifdef SPLATT_USE_MPI
  {"distribute", 'd', "DIM", 0, "MPI: dimension of data distribution "
                                 "(default: 3)"},
#endif
  { 0 }
};


typedef struct
{
  char * ifname;   /** file that we read the tensor from */
  int write;       /** do we write output to file? */
  double * opts;   /** splatt_cpd options */
  idx_t nfactors;
#ifdef SPLATT_USE_MPI
  int distribution;
  int mpi_dims[MAX_NMODES];
#endif
} cpd_cmd_args;


/**
* @brief Fill a cpd_opts struct with default values.
*
* @param args The cpd_opts struct to fill.
*/
void default_cpd_opts(
  cpd_cmd_args * args)
{
  args->opts = splatt_default_opts();
  args->ifname    = NULL;
  args->write     = DEFAULT_WRITE;
  args->nfactors  = DEFAULT_NFACTORS;

#ifdef SPLATT_USE_MPI
  args->distribution = DEFAULT_MPI_DISTRIBUTION;
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    args->mpi_dims[m] = 1;
  }
#endif
}



static error_t parse_cpd_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  cpd_cmd_args * args = state->input;
  char * buf;
  int cnt = 0;

  /* -i=50 should also work... */
  if(arg != NULL && arg[0] == '=') {
    ++arg;
  }

  switch(key) {
  case 'i':
    args->opts[SPLATT_OPTION_NITER] = (double) atoi(arg);
    break;
  case TT_TOL:
    args->opts[SPLATT_OPTION_TOLERANCE] = atof(arg);
    break;
  case 't':
    args->opts[SPLATT_OPTION_NTHREADS] = (double) atoi(arg);
    break;
  case 'v':
    timer_inc_verbose();
    args->opts[SPLATT_OPTION_VERBOSITY] += 1;
    break;
  case TT_TILE:
    args->opts[SPLATT_OPTION_TILE] = SPLATT_SYNCTILE;
    break;
  case TT_NOWRITE:
    args->write = 0;
    break;
  case 'r':
    args->nfactors = atoi(arg);
    break;
#ifdef SPLATT_USE_MPI
  case 'd':
    buf = strtok(arg, "x");
    while(buf != NULL) {
      args->mpi_dims[cnt++] = atoi(buf);
      buf = strtok(NULL, "x");
    }
    args->distribution = cnt;
    break;
#endif

  case ARGP_KEY_ARG:
    if(args->ifname != NULL) {
      argp_usage(state);
      break;
    }
    args->ifname = arg;
    break;
  case ARGP_KEY_END:
    if(args->ifname == NULL) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}

static struct argp cpd_argp =
  {cpd_options, parse_cpd_opt, cpd_args_doc, cpd_doc};


#ifdef SPLATT_USE_MPI
void splatt_mpi_cpd_cmd(
  int argc,
  char ** argv)
{
  /* assign defaults and parse arguments */
  cpd_cmd_args args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  sptensor_t * tt = NULL;

  rank_info rinfo;
  MPI_Comm_rank(MPI_COMM_WORLD, &rinfo.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rinfo.npes);

  rinfo.distribution = args.distribution;
  for(int d=0; d < args.distribution; ++d) {
    rinfo.dims_3d[d] = SS_MAX(args.mpi_dims[d], 1);
  }

  if(rinfo.rank == 0) {
    print_header();
  }

  tt = mpi_tt_read(args.ifname, &rinfo);
  ftensor_t * ft = (ftensor_t *) malloc(tt->nmodes * sizeof(ftensor_t));

  /* In the default setting, mpi_tt_read will set rinfo distribution.
   * Copy that back into args. TODO: make this less dumb. */
  args.distribution = rinfo.distribution;
  for(int m=0; m < args.distribution; ++m) {
    args.mpi_dims[m] = rinfo.dims_3d[m];
  }

  /* print stats */
  if(rinfo.rank == 0) {
    mpi_global_stats(tt, &rinfo, args.ifname);
  }

  /* determine matrix distribution - this also calls tt_remove_empty() */
  permutation_t * perm = mpi_distribute_mats(&rinfo, tt, rinfo.distribution);

  /* 1D and 2D distributions require filtering because tt has nonzeros that
   * don't belong in each ftensor */
  if(rinfo.distribution == 1) {
    /* compress tensor to own local coordinate system */
    tt_remove_empty(tt);

    sptensor_t * tt_filtered = tt_alloc(tt->nnz, tt->nmodes);
    for(idx_t m=0; m < tt->nmodes; ++m) {
      /* tt has more nonzeros than any of the modes actually need, so we need
       * to filter them first. */
      mpi_filter_tt_1d(m, tt, tt_filtered, rinfo.mat_start[m],
          rinfo.mat_end[m]);
      assert(tt_filtered->dims[m] == rinfo.mat_end[m] - rinfo.mat_start[m]);

      mpi_find_owned(tt, m, &rinfo);
      mpi_compute_ineed(&rinfo, tt, m, args.nfactors, 1);

      ften_alloc(ft + m, tt_filtered, m, (int) args.opts[SPLATT_OPTION_TILE]);
      /* sanity check on nnz */
      idx_t totnnz;
      MPI_Reduce(&ft[m].nnz, &totnnz, 1, MPI_DOUBLE, MPI_SUM, 0,
          MPI_COMM_WORLD);
      if(rinfo.rank == 0) {
        assert(totnnz == rinfo.global_nnz);
      }
    } /* foreach mode */

    tt_free(tt_filtered);

  /* 3D distribution is simpler */
  } else {
    /* compress tensor to own local coordinate system */
    tt_remove_empty(tt);

    for(idx_t m=0; m < tt->nmodes; ++m) {
      /* index into local tensor to grab owned rows */
      mpi_find_owned(tt, m, &rinfo);
      /* determine isend and ineed lists */
      mpi_compute_ineed(&rinfo, tt, m, args.nfactors, 3);
      /* fill each ftensor */
      ften_alloc(ft + m, tt, m, (int) args.opts[SPLATT_OPTION_TILE]);
    }
  } /* end 3D distribution */

  mpi_rank_stats(tt, &rinfo);

  idx_t const nmodes = tt->nmodes;
  tt_free(tt);

  /* allocate / initialize matrices */
  idx_t max_dim = 0;
  /* M, the result matrix is stored at mats[MAX_NMODES] */
  matrix_t * mats[MAX_NMODES+1];
  matrix_t * globmats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    /* ft[:] have different dimensionalities for 1/2D but ft[m+1] is guaranteed
     * to have the full dimensionality */
    mats[m] = mat_rand(ft[(m+1) % nmodes].dims[m], args.nfactors);
    max_dim = SS_MAX(max_dim, ft[(m+1) % nmodes].dims[m]);

    /* for actual factor matrix */
    globmats[m] = mat_rand(rinfo.mat_end[m] - rinfo.mat_start[m],
        args.nfactors);
  }
  mats[MAX_NMODES] = mat_alloc(max_dim, args.nfactors);

  val_t * lambda = (val_t *) malloc(args.nfactors * sizeof(val_t));

  /* find total storage */
  unsigned long fbytes = 0;
  unsigned long mbytes = 0;
  for(idx_t m=0; m < nmodes; ++m) {
    fbytes += ften_storage(&(ft[m]));
    mbytes += ft[m].dims[m] * args.nfactors * sizeof(val_t);
  }
  /* get storage across all nodes */
  if(rinfo.rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &fbytes, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
        MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&fbytes, NULL, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  if(rinfo.rank == 0) {
    printf("Factoring "
           "------------------------------------------------------\n");
    printf("NFACTORS=%"SPLATT_PF_IDX" MAXITS=%"SPLATT_PF_IDX" TOL=%0.1e ",
        args.nfactors,
        (idx_t) args.opts[SPLATT_OPTION_NITER],
        args.opts[SPLATT_OPTION_TOLERANCE]);
    printf("RANKS=%d ", rinfo.npes);
    printf("THREADS=%"SPLATT_PF_IDX" ", (idx_t) args.opts[SPLATT_OPTION_NTHREADS]);
    if((int) args.opts[SPLATT_OPTION_TILE] != SPLATT_NOTILE) {
      printf("TILE=%"SPLATT_PF_IDX"x%"SPLATT_PF_IDX"x%"SPLATT_PF_IDX"\n",
        TILE_SIZES[0], TILE_SIZES[1], TILE_SIZES[2]);
    } else {
      printf("TILE=NO\n");
    }
    char * fstorage = bytes_str(fbytes);
    char * mstorage = bytes_str(mbytes);
    printf("CSF-STORAGE=%s FACTOR-STORAGE=%s", fstorage, mstorage);
    free(fstorage);
    free(mstorage);
    printf("\n\n");
  }

  /* do the factorization! */
  //cpd_als(ft, mats, globmats, lambda, args.nfactors, &rinfo, args.opts);

  /* free up the ftensor allocations */
  for(idx_t m=0; m < nmodes; ++m) {
    ften_free(&(ft[m]));
  }
  free(ft);

  /* write output */
  if(args.write == 1) {
    mpi_write_mats(globmats, perm, &rinfo, "mode", nmodes);
    vec_write(lambda, args.nfactors, "lambda.mat");
  }

  /* free factor matrix allocations */
  for(idx_t m=0;m < nmodes; ++m) {
    mat_free(mats[m]);
    mat_free(globmats[m]);
  }
  mat_free(mats[MAX_NMODES]);
  free(lambda);
  free(args.opts);

  perm_free(perm);
  rank_free(rinfo, nmodes);
}
#endif


/******************************************************************************
 * SPLATT-CPD
 *****************************************************************************/
void splatt_cpd_cmd(
  int argc,
  char ** argv)
{
  /* assign defaults and parse arguments */
  cpd_cmd_args args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  sptensor_t * tt = NULL;

  rank_info rinfo;
  rinfo.rank = 0;
  print_header();

  tt = tt_read(args.ifname);
  if(tt == NULL) {
    return;
  }
  tt_remove_empty(tt);
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  ftensor_t * ft = (ftensor_t *) malloc(tt->nmodes * sizeof(ftensor_t));

  /* fill each ftensor */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    ften_alloc(ft + m, tt, m, (int) args.opts[SPLATT_OPTION_TILE]);
  }

  splatt_csf * csf = splatt_csf_alloc(tt, args.opts);
  splatt_csf_free(csf, args.opts);

  idx_t const nmodes = tt->nmodes;
  tt_free(tt);

  /* allocate / initialize matrices */
  idx_t max_dim = 0;
  /* M, the result matrix is stored at mats[MAX_NMODES] */
  matrix_t * mats[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    /* ft[:] have different dimensionalities for 1/2D but ft[m+1] is guaranteed
     * to have the full dimensionality */
    mats[m] = mat_rand(ft[(m+1) % nmodes].dims[m], args.nfactors);
    max_dim = SS_MAX(max_dim, ft[(m+1) % nmodes].dims[m]);
  }
  mats[MAX_NMODES] = mat_alloc(max_dim, args.nfactors);

  val_t * lambda = (val_t *) malloc(args.nfactors * sizeof(val_t));

  /* find total storage */
  unsigned long fbytes = 0;
  unsigned long mbytes = 0;
  for(idx_t m=0; m < nmodes; ++m) {
    fbytes += ften_storage(&(ft[m]));
    mbytes += ft[m].dims[m] * args.nfactors * sizeof(val_t);
  }

  printf("Factoring "
         "------------------------------------------------------\n");
  printf("NFACTORS=%"SPLATT_PF_IDX" MAXITS=%"SPLATT_PF_IDX" TOL=%0.1e ",
      args.nfactors,
      (idx_t) args.opts[SPLATT_OPTION_NITER],
      args.opts[SPLATT_OPTION_TOLERANCE]);
  printf("THREADS=%"SPLATT_PF_IDX" ", (idx_t) args.opts[SPLATT_OPTION_NTHREADS]);
  if((int) args.opts[SPLATT_OPTION_TILE] != SPLATT_NOTILE) {
    printf("TILE=%"SPLATT_PF_IDX"x%"SPLATT_PF_IDX"x%"SPLATT_PF_IDX"\n",
      TILE_SIZES[0], TILE_SIZES[1], TILE_SIZES[2]);
  } else {
    printf("TILE=NO\n");
  }
  char * fstorage = bytes_str(fbytes);
  char * mstorage = bytes_str(mbytes);
  printf("CSF-STORAGE=%s FACTOR-STORAGE=%s", fstorage, mstorage);
  free(fstorage);
  free(mstorage);
  printf("\n\n");

  /* do the factorization! */
  //cpd_als(ft, mats, mats, lambda, args.nfactors, &rinfo, args.opts);

  /* free up the ftensor allocations */
  for(idx_t m=0; m < nmodes; ++m) {
    ften_free(&ft[m]);
  }
  free(ft);
  free(args.opts);

  /* write output */
  if(args.write == 1) {
    mat_write(mats[0], "mode1.mat");
    mat_write(mats[1], "mode2.mat");
    mat_write(mats[2], "mode3.mat");
    vec_write(lambda, args.nfactors, "lambda.mat");
  }

  /* free factor matrix allocations */
  for(idx_t m=0;m < nmodes; ++m) {
    mat_free(mats[m]);
  }
  mat_free(mats[MAX_NMODES]);
  free(lambda);
}

