
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../stats.h"
#include "../cpd.h"

#include "../mpi.h"


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
  srand(rank);

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
  /* compress tensor to own local coordinate system */
  tt_remove_empty(tt);

  /* determine isend and ineed lists */
  mpi_compute_ineed(&rinfo, tt);

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

  if(rank == 0) {
    printf("\n");
    printf("Factoring "
           "------------------------------------------------------\n");
    printf("RANK=%"SS_IDX" MAXITS=%"SS_IDX"\n", args.rank, args.niters);
  }

  /* do the actual factorization */
  mpi_cpd(tt, mats, globmats, &rinfo, &args);

  mat_free(mats[MAX_NMODES]);
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mat_free(mats[m]);
    mat_free(globmats[m]);
  }

  rank_free(rinfo, tt->nmodes);
  perm_free(perm);
  tt_free(tt);
}

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

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size > 1) {
    __par_cpd(argc, argv);
    return;
  }

  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  print_header();

  sptensor_t * tt = tt_read(args.ifname);
  stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);

  /* M, the result matrix is stored at mats[MAX_NMODES] */
  idx_t max_dim = 0;
  matrix_t * mats[MAX_NMODES+1];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mats[m] = mat_rand(tt->dims[m], args.rank);
    if(tt->dims[m] > max_dim) {
      max_dim = tt->dims[m];
    }
  }
  mats[MAX_NMODES] = mat_alloc(max_dim, args.rank);

  printf("Factoring ------------------------------------------------------\n");
  printf("RANK=%"SS_IDX" MAXITS=%"SS_IDX"\n", args.rank, args.niters);

  cpd(tt, mats, &args);

  for(idx_t m=0;m < tt->nmodes; ++m) {
    mat_free(mats[m]);
  }
  mat_free(mats[MAX_NMODES]);

  tt_free(tt);
}

