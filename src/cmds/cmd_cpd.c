
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../tile.h"
#include "../stats.h"
#include "../cpd.h"
#include "../splatt_mpi.h"


/******************************************************************************
 * SPLATT CPD
 *****************************************************************************/
static char cpd_args_doc[] = "TENSOR";
static char cpd_doc[] =
  "splatt-cpd -- compute the CPD of a sparse tensor\n\n";

#define TT_TILE 255
static struct argp_option cpd_options[] = {
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
  switch(key) {
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

#if 0
/* XXX: this is temporary */
static void __par_cpd(
  int argc,
  char ** argv)
{
  cpd_opts args;
  args.ifname = NULL;
  args.niters = 5;
  args.rank = 10;
  args.nthreads = 1;
  args.tile = 0;
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(rank == 0) {
    print_header();
  }

  /* XXX: this should probably be improved... */
  srand(time(NULL) * rank);

  rank_info rinfo;
  mpi_setup_comms(&rinfo);
  sptensor_t * tt = mpi_tt_read(args.ifname, &rinfo);

  if(rank == 0) {
    printf("global dims:\t\t%8lu %8lu %8lu %8lu\n",
      rinfo.global_dims[0],
      rinfo.global_dims[1],
      rinfo.global_dims[2],
      rinfo.global_nnz);
    printf("max dims:\t\t%8lu %8lu %8lu\n",
      rinfo.global_dims[0] / rinfo.np13,
      rinfo.global_dims[1] / rinfo.np13,
      rinfo.global_dims[2] / rinfo.np13);
    printf("rank dims:\t\t%8lu %8lu %8lu %8lu\n",
      rinfo.global_dims[0] / size,
      rinfo.global_dims[1] / size,
      rinfo.global_dims[2] / size,
      rinfo.global_nnz / size);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* determine matrix distribution */
  permutation_t * perm = mpi_distribute_mats(&rinfo, tt);

  /* determine isend and ineed lists */
  mpi_compute_ineed(&rinfo, tt, args.rank);

#if 0
  printf("%d:\t\t\t%8lu %8lu %8lu %8lu\n", rank, tt->dims[0], tt->dims[1],
      tt->dims[2], tt->nnz);
  mpi_send_recv_stats(&rinfo, tt);
#endif

  /* allocate / initialize matrices */
  matrix_t * mats[MAX_NMODES+1];
  matrix_t * globmats[MAX_NMODES];
  idx_t max_dim = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    /* for multiplication */
    mats[m] = mat_alloc(tt->dims[m], args.rank);

    /* for actual factor matrix */
    globmats[m] = mat_rand(rinfo.mat_end[m] - rinfo.mat_start[m], args.rank);

    if(tt->dims[m] > max_dim) {
      max_dim = tt->dims[m];
    }
  }
  mats[MAX_NMODES] = mat_alloc(max_dim, args.rank);

  val_t * lambda = (val_t *) malloc(args.rank * sizeof(val_t));

  if(rank == 0) {
    printf("\n");
    printf("Factoring "
           "------------------------------------------------------\n");
    printf("RANK=%"SS_IDX" MAXITS=%"SS_IDX" THREADS=%"SS_IDX"\n",
        args.rank, args.niters, args.nthreads);
  }

  /* do the actual factorization */
  cpd(tt, mats, globmats, lambda, &rinfo, &args);

  idx_t const nmodes = tt->nmodes;
  tt_free(tt);
  mat_free(mats[MAX_NMODES]);
  for(idx_t m=0; m < nmodes; ++m) {
    mat_free(mats[m]);
  }
  free(lambda);

  /* write output */
  //mpi_write_mats(globmats, perm, &rinfo, "test", nmodes);

  for(idx_t m=0; m < nmodes; ++m) {
    mat_free(globmats[m]);
  }
  perm_free(perm);
  rank_free(rinfo, nmodes);
}
#endif /* USE_MPI */

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

  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  rank_info rinfo;
  rinfo.rank = 0;

  sptensor_t * tt = NULL;

#ifdef USE_MPI
  mpi_setup_comms(&rinfo);
  if(rinfo.npes == 1) {
    fprintf(stderr, "SPLATT: I was configured with MPI support. Please re-run\n"
                    "        with > 1 ranks or recompile without --mpi.\n");
    abort();
  }
  tt = mpi_tt_read(args.ifname, &rinfo);
#else
  tt = tt_read(args.ifname);
#endif

  if(rinfo.rank == 0) {
    print_header();
    stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);
  }

#ifdef USE_MPI
  /* determine matrix distribution */
  permutation_t * perm = mpi_distribute_mats(&rinfo, tt);

  /* determine isend and ineed lists */
  mpi_compute_ineed(&rinfo, tt, args.rank);
#endif

  /* allocate / initialize matrices */
  idx_t max_dim = 0;
  /* M, the result matrix is stored at mats[MAX_NMODES] */
  matrix_t * mats[MAX_NMODES+1];
  matrix_t * globmats[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mats[m] = mat_rand(tt->dims[m], args.rank);
    if(tt->dims[m] > max_dim) {
      max_dim = tt->dims[m];
    }
#ifdef USE_MPI
    /* for actual factor matrix */
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
#ifdef USE_MPI
    printf("RANKS=%d ", rinfo.npes);
#endif
    printf("THREADS=%"SS_IDX" ", args.nthreads);
    if(args.tile == 1) {
      printf("TILE=%"SS_IDX"x%"SS_IDX"x%"SS_IDX" ",
        TILE_SIZES[0], TILE_SIZES[1], TILE_SIZES[2]);
    } else {
      printf("TILE=NO ");
    }
    printf("\n");
  }

  /* do the factorization! */
  cpd(tt, mats, globmats, lambda, &rinfo, &args);

  idx_t const nmodes = tt->nmodes;
  tt_free(tt);
  for(idx_t m=0;m < tt->nmodes; ++m) {
    mat_free(mats[m]);
#ifdef USE_MPI
    mat_free(globmats[m]);
#endif
  }
  mat_free(mats[MAX_NMODES]);
  free(lambda);

#ifdef USE_MPI
  /* write output */
  //mpi_write_mats(globmats, perm, &rinfo, "test", nmodes);
  perm_free(perm);
  rank_free(rinfo, nmodes);
#endif
}

