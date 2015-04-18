
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


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


#ifdef SPLATT_USE_MPI
/**
* @brief Copy global information into local tt, print statistics, and
*        restore local information.
*
* @param tt The tensor to hold global information.
* @param rinfo Global tensor information.
*/
static void __mpi_global_stats(
  sptensor_t * const tt,
  rank_info * const rinfo,
  cpd_opts const * const args)
{
  idx_t * tmpdims = tt->dims;
  idx_t tmpnnz = tt->nnz;
  tt->dims = rinfo->global_dims;
  tt->nnz = rinfo->global_nnz;

  /* print stats */
  stats_tt(tt, args->ifname, STATS_BASIC, 0, NULL);

  /* restore local stats */
  tt->dims = tmpdims;
  tt->nnz = tmpnnz;
}


static void __mpi_rank_stats(
  sptensor_t const * const tt,
  rank_info const * const rinfo,
  cpd_opts const * const args)
{
  idx_t totnnz = 0;
  idx_t maxnnz = 0;
  idx_t totvolume = 0;
  idx_t maxvolume = 0;
  idx_t volume = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    /* if a layer has > 1 rank there is a necessary reduction step too */
    if(rinfo->layer_size[m] > 1) {
      volume += 2 * (rinfo->nlocal2nbr[m] + rinfo->nnbr2globs[m]);
    } else {
      volume += rinfo->nlocal2nbr[m] + rinfo->nnbr2globs[m];
    }
  }
  MPI_Reduce(&volume, &totvolume, 1, SS_MPI_IDX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&volume, &maxvolume, 1, SS_MPI_IDX, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&tt->nnz, &totnnz, 1, SS_MPI_IDX, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&tt->nnz, &maxnnz, 1, SS_MPI_IDX, MPI_MAX, 0, MPI_COMM_WORLD);

  if(rinfo->rank == 0) {
    printf("MPI information ------------------------------------------------\n");
    printf("DISTRIBUTION=%"SS_IDX"D ", args->distribution);
    printf("DIMS=%dx%dx%d\n", rinfo->dims_3d[0], rinfo->dims_3d[1],
        rinfo->dims_3d[2]);
    idx_t avgvolume = totvolume / rinfo->npes;

    idx_t const avgnnz = totnnz / rinfo->npes;
    double nnzimbalance = 100. * ((double)(maxnnz - avgnnz) / (double)maxnnz);
    double volimbalance = 100. * ((double)(maxvolume - avgvolume) /
        (double)maxvolume);
    printf("AVG NNZ=%"SS_IDX"\nMAX NNZ=%"SS_IDX"  (%0.2f%% diff)\n",
        avgnnz, maxnnz, nnzimbalance);
    printf("AVG COMMUNICATION VOL=%"SS_IDX"\nMAX COMMUNICATION VOL=%"SS_IDX"  "
        "(%0.2f%% diff)\n", avgvolume, maxvolume, volimbalance);
    printf("\n");
  }
}
#endif


/******************************************************************************
 * SPLATT CPD
 *****************************************************************************/
static char cpd_args_doc[] = "TENSOR";
static char cpd_doc[] =
  "splatt-cpd -- compute the CPD of a sparse tensor\n\n";

#define TT_TILE 255
static struct argp_option cpd_options[] = {
  {"distribute", 'd', "DIM", 0, "MPI: dimension of data distribution "
                                 "(default: 3)"},
  {"iters", 'i', "NITERS", 0, "number of iterations to use (default: 5)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: 1)"},
  {"tile", TT_TILE, 0, 0, "use tiling during SPLATT"},
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
  switch(key) {
  case 'd':
    buf = strtok(arg, "x");
    while(buf != NULL) {
      args->mpi_dims[cnt++] = atoi(buf);
      buf = strtok(NULL, "x");
    }
    args->distribution = cnt;
    break;
  case 'i':
    args->niters = atoi(arg);
    break;
  case 'r':
    args->rank = atoi(arg);
    break;
  case 't':
    args->nthreads = atoi(arg);
    break;
  case TT_TILE:
    args->tile = 1;
    break;

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

void splatt_cpd(
  int argc,
  char ** argv)
{
  cpd_opts args;
  args.ifname = NULL;
  args.niters = 5;
  args.rank = 10;
  args.nthreads = 1;
  args.tile = 0;
  args.distribution = 3;
  args.mpi_dims[0] = args.mpi_dims[1] = args.mpi_dims[2] = 1;

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
  if(rinfo.npes == 1) {
    fprintf(stderr, "SPLATT: I was configured with MPI support. Please re-run\n"
                    "        with > 1 ranks or recompile without MPI.\n");
    abort();
  }
  tt = mpi_tt_read(args.ifname, &rinfo);

  /* print stats */
  if(rinfo.rank == 0) {
    __mpi_global_stats(tt, &rinfo, &args);
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

  __mpi_rank_stats(tt, &rinfo, &args);
#else
  tt = tt_read(args.ifname);
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  tt_remove_empty(tt);

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
    /* ft[:] have different dimensionalities for 1/2D and ft[m+1] is guaranteed
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

  if(rinfo.rank == 0) {
    printf("Factoring "
           "------------------------------------------------------\n");
    printf("NFACTORS=%"SS_IDX" MAXITS=%"SS_IDX" ", args.rank, args.niters);
#ifdef SPLATT_USE_MPI
    printf("RANKS=%d ", rinfo.npes);
#endif
    printf("THREADS=%"SS_IDX" ", args.nthreads);
    if(args.tile == 1) {
      printf("TILE=%"SS_IDX"x%"SS_IDX"x%"SS_IDX" ",
        TILE_SIZES[0], TILE_SIZES[1], TILE_SIZES[2]);
    } else {
      printf("TILE=NO ");
    }
    printf("\n\n");
  }

  /* do the factorization! */
  cpd(ft, mats, globmats, lambda, &rinfo, &args);

  for(idx_t m=0;m < nmodes; ++m) {
    ften_free(ft[m]);
    mat_free(mats[m]);
#ifdef SPLATT_USE_MPI
    mat_free(globmats[m]);
#endif
  }
  mat_free(mats[MAX_NMODES]);
  free(lambda);

#ifdef SPLATT_USE_MPI
  /* write output */
  //mpi_write_mats(globmats, perm, &rinfo, "test", nmodes);
  perm_free(perm);
  rank_free(rinfo, nmodes);
#endif
}

