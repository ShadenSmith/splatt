
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
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: 1)"},
  {"tile", TT_TILE, 0, 0, "use tiling during SPLATT"},
  {"nowrite", TT_NOWRITE, 0, 0, "do not write output to file (default: WRITE)"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
#ifdef SPLATT_USE_MPI
  {"distribute", 'd', "DIM", 0, "MPI: dimension of data distribution "
                                 "(default: 3)"},
#endif
  { 0 }
};


static error_t parse_cpd_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  cpd_opts *args = state->input;
  char * buf;
  int cnt = 0;

  /* -i=50 should also work... */
  if(arg != NULL && arg[0] == '=') {
    ++arg;
  }

  switch(key) {
  case TT_NOWRITE:
    args->write = 0;
    break;
  case 'i':
    args->niters = atoi(arg);
    break;
  case TT_TOL:
    args->tol = atof(arg);
    break;
  case 'r':
    args->rank = atoi(arg);
    break;
  case 't':
    args->nthreads = atoi(arg);
    break;
  case 'v':
    timer_lvl = TIMER_LVL2;
    break;
  case TT_TILE:
    args->tile = 1;
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



/******************************************************************************
 * SPLATT-CPD
 *****************************************************************************/
void splatt_cpd_cmd(
  int argc,
  char ** argv)
{
  /* assign defaults and parse arguments */
  cpd_opts args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  sptensor_t * tt = NULL;
  ftensor_t * ft[MAX_NMODES];

  rank_info rinfo;
#ifdef SPLATT_USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rinfo.rank);
  for(idx_t d=0; d < args.distribution; ++d) {
    rinfo.dims_3d[d] = SS_MAX(args.mpi_dims[d], 1);
  }
#else
  rinfo.rank = 0;
#endif

  if(rinfo.rank == 0) {
    print_header();
  }

#ifdef SPLATT_USE_MPI
  mpi_setup_comms(&rinfo, args.distribution);
  tt = mpi_tt_read(args.ifname, &rinfo);

  /* print stats */
  if(rinfo.rank == 0) {
    mpi_global_stats(tt, &rinfo, &args);
  }

  /* determine matrix distribution - this also calls tt_remove_empty() */
  permutation_t * perm = mpi_distribute_mats(&rinfo, tt, args.distribution);

  /* 1D and 2D distributions require filtering because tt has nonzeros that
   * don't belong in each ftensor */
  if(args.distribution == 1) {
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
      mpi_compute_ineed(&rinfo, tt, m, args.rank, 1);

      ft[m] = ften_alloc(tt_filtered, m, args.tile);
      /* sanity check on nnz */
      idx_t totnnz;
      MPI_Reduce(&ft[m]->nnz, &totnnz, 1, MPI_DOUBLE, MPI_SUM, 0,
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
      mpi_compute_ineed(&rinfo, tt, m, args.rank, 3);
      /* fill each ftensor */
      ft[m] = ften_alloc(tt, m, args.tile);
    }
  } /* end 3D distribution */

  mpi_rank_stats(tt, &rinfo, &args);
#else
  tt = tt_read(args.ifname);
  tt_remove_empty(tt);
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  /* fill each ftensor */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    ft[m] = ften_alloc(tt, m, args.tile);
  }
#endif

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
    mats[m] = mat_rand(ft[(m+1) % nmodes]->dims[m], args.rank);
    max_dim = SS_MAX(max_dim, ft[(m+1) % nmodes]->dims[m]);

    /* for actual factor matrix */
#ifdef SPLATT_USE_MPI
    globmats[m] = mat_rand(rinfo.mat_end[m] - rinfo.mat_start[m], args.rank);
#else
    globmats[m] = mats[m];
#endif
  }
  mats[MAX_NMODES] = mat_alloc(max_dim, args.rank);

  val_t * lambda = (val_t *) malloc(args.rank * sizeof(val_t));

  /* find total storage */
  unsigned long fbytes = 0;
  unsigned long mbytes = 0;
  for(idx_t m=0; m < nmodes; ++m) {
    fbytes += ften_storage(ft[m]);
    mbytes += ft[m]->dims[m] * args.rank * sizeof(val_t);
  }
#ifdef SPLATT_USE_MPI
  /* get storage across all nodes */
  if(rinfo.rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &fbytes, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
        MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&fbytes, NULL, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  idx_t nfibs = 0;
  for(idx_t m=0; m < nmodes; ++m) {
    nfibs += ft[m]->nfibs;
    printf("m: %lu, %lu\n", m, ft[m]->nfibs);
  }
  printf("nfibs: %lu\n", nfibs);

  if(rinfo.rank == 0) {
    printf("Factoring "
           "------------------------------------------------------\n");
    printf("NFACTORS=%"SS_IDX" MAXITS=%"SS_IDX" TOL=%0.1e ", args.rank,
        args.niters, args.tol);
#ifdef SPLATT_USE_MPI
    printf("RANKS=%d ", rinfo.npes);
#endif
    printf("THREADS=%"SS_IDX" ", args.nthreads);
    if(args.tile == 1) {
      printf("TILE=%"SS_IDX"x%"SS_IDX"x%"SS_IDX"\n",
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
  cpd_als(ft, mats, globmats, lambda, &rinfo, &args);

  /* free up the ftensor allocations */
  for(idx_t m=0; m < nmodes; ++m) {
    ften_free(ft[m]);
  }

  /* write output */
  if(args.write == 1) {
#ifndef SPLATT_USE_MPI
    mat_write(globmats[0], "mode1.mat");
    mat_write(globmats[1], "mode2.mat");
    mat_write(globmats[2], "mode3.mat");
#else
    mpi_write_mats(globmats, perm, &rinfo, "mode", nmodes);
#endif
    vec_write(lambda, args.rank, "lambda.mat");
  }

  /* free factor matrix allocations */
  for(idx_t m=0;m < nmodes; ++m) {
    mat_free(mats[m]);
#ifdef SPLATT_USE_MPI
    mat_free(globmats[m]);
#endif
  }
  mat_free(mats[MAX_NMODES]);
  free(lambda);

#ifdef SPLATT_USE_MPI
  perm_free(perm);
  rank_free(rinfo, nmodes);
#endif
}

