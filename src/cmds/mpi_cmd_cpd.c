#ifdef SPLATT_USE_MPI

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../stats.h"
#include "../cpd.h"
#include "../splatt_mpi.h"
#include "../util.h"


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
  {"distribute", 'd', "DIM", 0, "MPI: dimension of data distribution "
                                 "(default: 3)"},
  {"partition", 'p', "FILE", 0, "MPI: partitioning for fine-grained"},
  { 0 }
};


typedef struct
{
  char * ifname;   /** file that we read the tensor from */
  int write;       /** do we write output to file? */
  double * opts;   /** splatt_cpd options */
  idx_t nfactors;
  splatt_decomp_type decomp;
  int mpi_dims[MAX_NMODES];
  char * pfname;   /** file that we read the partitioning from */
} cpd_cmd_args;


/**
* @brief Fill a cpd_opts struct with default values.
*
* @param args The cpd_opts struct to fill.
*/
static void default_cpd_opts(
  cpd_cmd_args * args)
{
  args->opts = splatt_default_opts();
  args->ifname    = NULL;
  args->pfname    = NULL;
  args->write     = DEFAULT_WRITE;
  args->nfactors  = DEFAULT_NFACTORS;

  args->decomp = DEFAULT_MPI_DISTRIBUTION;
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    args->mpi_dims[m] = 1;
  }
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
    args->opts[SPLATT_OPTION_VERBOSITY] += 1;
    break;
  case TT_TILE:
    args->opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
    break;
  case TT_NOWRITE:
    args->write = 0;
    break;
  case 'r':
    args->nfactors = atoi(arg);
    break;
  case 'd':
    /* fine-grained decomp */
    if(arg[0] == 'f') {
      args->opts[SPLATT_OPTION_DECOMP] = SPLATT_DECOMP_FINE;
      args->decomp = SPLATT_DECOMP_FINE;
      break;
    }
    buf = strtok(arg, "x");
    while(buf != NULL) {
      args->mpi_dims[cnt++] = atoi(buf);
      buf = strtok(NULL, "x");
    }
    if(cnt == 1) {
      args->opts[SPLATT_OPTION_DECOMP] = SPLATT_DECOMP_COARSE;
      args->decomp = SPLATT_DECOMP_COARSE;
    } else {
      args->decomp = SPLATT_DECOMP_MEDIUM;
    }
    break;

  case 'p':
    args->pfname = arg;
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



/******************************************************************************
 * SPLATT-CPD
 *****************************************************************************/

void splatt_mpi_cpd_cmd(
  int argc,
  char ** argv)
{
  /* assign defaults and parse arguments */
  cpd_cmd_args args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  sptensor_t * tt  = NULL;
  splatt_csf * csf = NULL;

  rank_info rinfo;
  MPI_Comm_rank(MPI_COMM_WORLD, &rinfo.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rinfo.npes);

  rinfo.decomp = args.decomp;
  for(int d=0; d < MAX_NMODES; ++d) {
    rinfo.dims_3d[d] = SS_MAX(args.mpi_dims[d], 1);
  }

  if(rinfo.rank == 0) {
    print_header();
  }

  tt = mpi_tt_read(args.ifname, args.pfname, &rinfo);

  /* In the default setting, mpi_tt_read will set rinfo distribution.
   * Copy that back into args. TODO: make this less dumb. */
  args.decomp = rinfo.decomp;
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    args.mpi_dims[m] = rinfo.dims_3d[m];
  }

  /* print stats */
  if(rinfo.rank == 0) {
    mpi_global_stats(tt, &rinfo, args.ifname);
  }

  /* determine matrix distribution - this also calls tt_remove_empty() */
  permutation_t * perm = mpi_distribute_mats(&rinfo, tt, rinfo.decomp);

  /* 1D distributions require filtering because tt has nonzeros that
   * don't belong in each ftensor */
  if(rinfo.decomp == SPLATT_DECOMP_COARSE) {
    /* XXX  TODO */
    csf = malloc(tt->nmodes * sizeof(*csf));
    /* compress tensor to own local coordinate system */
    tt_remove_empty(tt);

    /* coarse-grained forces us to use ALLMODE. override default opts. */
    args.opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE;

    sptensor_t * tt_filtered = tt_alloc(tt->nnz, tt->nmodes);
    for(idx_t m=0; m < tt->nmodes; ++m) {

      /* tt has more nonzeros than any of the modes actually need, so we need
       * to filter them first. */
      mpi_filter_tt_1d(m, tt, tt_filtered, rinfo.mat_start[m],
          rinfo.mat_end[m]);
      assert(tt_filtered->dims[m] == rinfo.mat_end[m] - rinfo.mat_start[m]);

      mpi_cpy_indmap(tt_filtered, &rinfo, m);

      mpi_find_owned(tt, m, &rinfo);
      mpi_compute_ineed(&rinfo, tt, m, args.nfactors, 1);

      /* fill csf[m] */
      csf_alloc_mode(tt_filtered, CSF_SORTED_MINUSONE, m, csf+m, args.opts);

      /* sanity check on nnz */
      idx_t totnnz;
      MPI_Reduce(&(csf[m].nnz), &totnnz, 1, MPI_DOUBLE, MPI_SUM, 0,
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
      mpi_cpy_indmap(tt, &rinfo, m);
    }

    /* create CSF tensor */
    csf = splatt_csf_alloc(tt, args.opts);

    for(idx_t m=0; m < tt->nmodes; ++m) {
      /* index into local tensor to grab owned rows */
      mpi_find_owned(tt, m, &rinfo);
      /* determine isend and ineed lists */
      mpi_compute_ineed(&rinfo, tt, m, args.nfactors, 3);
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
    /* ft[:] have different dimensionalities for 1D but ft[m+1] is guaranteed
     * to have the full dimensionality
     * */
    idx_t dim = csf->dims[m];
    if(rinfo.decomp == SPLATT_DECOMP_COARSE) {
      dim = csf[(m+1)%nmodes].dims[m];
    }
    max_dim = SS_MAX(max_dim, dim);

    mats[m] = mat_rand(dim, args.nfactors);

    /* for actual factor matrix */
    globmats[m] = mat_rand(rinfo.mat_end[m] - rinfo.mat_start[m],
        args.nfactors);
  }
  mats[MAX_NMODES] = mat_alloc(max_dim, args.nfactors);

  val_t * lambda = (val_t *) malloc(args.nfactors * sizeof(val_t));

  mpi_cpd_stats(csf, args.nfactors, args.opts, &rinfo);

  /* do the factorization! */
  mpi_cpd_als_iterate(csf, mats, globmats, lambda, args.nfactors, &rinfo,
      args.opts);

  /* free up the ftensor allocations */
  splatt_csf_free(csf, args.opts);

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
