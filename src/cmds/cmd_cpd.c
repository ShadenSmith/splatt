
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
  switch(key) {
  case 'd':
    args->distribution = atoi(arg);
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

  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  rank_info rinfo;
  rinfo.rank = 0;

  sptensor_t * tt = NULL;

#ifdef SPLATT_USE_MPI
  mpi_setup_comms(&rinfo, args.distribution);
  if(rinfo.npes == 1) {
    fprintf(stderr, "SPLATT: I was configured with MPI support. Please re-run\n"
                    "        with > 1 ranks or recompile without MPI.\n");
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

#ifdef SPLATT_USE_MPI
  /* determine matrix distribution */
  permutation_t * perm = mpi_distribute_mats(&rinfo, tt);

  /* determine isend and ineed lists */
  mpi_compute_ineed(&rinfo, tt, args.rank);
#else
  tt_remove_empty(tt);
#endif

  ftensor_t * ft[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    ft[m] = ften_alloc(tt, m, args.tile);
  }
  tt_free(tt);

  /* allocate / initialize matrices */
  idx_t max_dim = 0;
  /* M, the result matrix is stored at mats[MAX_NMODES] */
  matrix_t * mats[MAX_NMODES+1];
  matrix_t * globmats[MAX_NMODES];
  for(idx_t m=0; m < ft[0]->nmodes; ++m) {
    mats[m] = mat_rand(ft[0]->dims[m], args.rank);
    if(ft[0]->dims[m] > max_dim) {
      max_dim = ft[0]->dims[m];
    }
#ifdef SPLATT_USE_MPI
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
    printf("\n");
  }

  /* do the factorization! */
  cpd(ft, mats, globmats, lambda, &rinfo, &args);

  idx_t const nmodes = ft[0]->nmodes;
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

