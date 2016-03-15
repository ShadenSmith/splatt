
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../completion/completion.h"
#include "../stats.h"
#include <omp.h>



/******************************************************************************
 * ARG PARSING
 *****************************************************************************/

static char gen_args_doc[] = "-i INPUT OUTPUT";
static char gen_doc[] =
  "splatt-generate -- Create a sparse tensor from a random CPD factorization.\n"
  "Available tensor generation are:\n"
  "  bootstrap\t\tuse the sparsity pattern of an existing tensor\n";


#define GEN_SEED 255
static struct argp_option gen_options[] = {
  {"input", 'i', "TENSOR", 0, "sparse tensor to bootstrap sparsity pattern"},
  {"rank", 'r', "RANK", 0, "rank (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"alg", 'a', "ALG", 0, "which generation algorithm to use (default: bootstrap)"},
  {"seed", GEN_SEED, "SEED", 0, "random seed (default: system time)"},
  {0}
};


typedef enum
{
  SPLATT_GEN_RAND,
  SPLATT_GEN_BOOTSTRAP,
  SPLATT_GEN_NALGS
} splatt_gen_type;

typedef struct
{
  char * name;
  splatt_gen_type which;
} gen_alg_map;

static gen_alg_map maps[] = {
  { "rand", SPLATT_GEN_RAND },
  { "bootstrap", SPLATT_GEN_BOOTSTRAP },
  { NULL,  SPLATT_GEN_NALGS }
};


/**
* @brief Parse a command into a splatt_gen_type (generation algorithm).
*
* @param arg The string to parse.
*
* @return The optimization algorithm to use. SPLATT_GEN_NALGS on error.
*/
static splatt_gen_type parse_gen_alg(
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
  return SPLATT_GEN_NALGS;
}



typedef struct
{
  char * ifname;
  char * ofname;

  bool set_seed;
  unsigned int seed;

  splatt_gen_type which_alg;

  idx_t nfactors;
  idx_t nthreads;
} gen_cmd_args;


/**
* @brief Fill the tensor generation arguments with some sane defaults.
*
* @param args The arguments to initialize.
*/
static void default_gen_opts(
    gen_cmd_args * const args)
{
  args->ifname = NULL;
  args->ofname = NULL;

  args->which_alg = SPLATT_GEN_BOOTSTRAP;
  args->nfactors = 10;
  args->nthreads = omp_get_max_threads();
  args->set_seed = false;
  args->seed = time(NULL);
}



static error_t parse_gen_opt(
    int key,
    char * arg,
    struct argp_state * state)
{
  gen_cmd_args * args = state->input;
  char * buf;
  int cnt = 0;

  /* -i=50 should also work... */
  if(arg != NULL && arg[0] == '=') {
    ++arg;
  }

  switch(key) {
  case 'i':
    if(args->ifname != NULL) {
      argp_usage(state);
      break;
    }
    args->ifname = arg;
    break;
  case 't':
    args->nthreads = strtoull(arg, &buf, 10);
    omp_set_num_threads(args->nthreads);
    break;
  case 'a':
    args->which_alg = parse_gen_alg(arg);
    if(args->which_alg == SPLATT_GEN_NALGS) {
      fprintf(stderr, "SPLATT: unknown generation algorithm '%s'.\n", arg);
      argp_usage(state);
    }
    break;
  case 'r':
    args->nfactors = strtoull(arg, &buf, 10);
    break;
  case GEN_SEED:
    args->seed = (unsigned int) atoi(arg);
    args->set_seed = true;
    break;

  case ARGP_KEY_ARG:
    if(args->ofname != NULL) {
      argp_usage(state);
      break;
    }
    args->ofname = arg;
    break;

  case ARGP_KEY_END:
    if(args->ofname == NULL) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}


/* tie it all together */
static struct argp gen_argp = {gen_options, parse_gen_opt, gen_args_doc, gen_doc};



/**
* @brief Generate a random sparse tensor whose sparsity structure is
*        bootstrapped from args->ifname.
*
* @param args Generation parameters.
*
* @return A random sparse tensor.
*/
static sptensor_t * p_rand_bootstrap(
    gen_cmd_args * args)
{
  /* load bootstrap tensor */
  sptensor_t * bootstrap = tt_read(args->ifname);
  stats_tt(bootstrap, args->ifname, STATS_BASIC, 0, NULL);
  val_t * const restrict vals = bootstrap->vals;

  /* random factorization */
  tc_model * model = tc_model_alloc(bootstrap, args->nfactors, SPLATT_TC_NALGS);


  #pragma omp parallel
  {
    val_t * buffer = splatt_malloc(args->nfactors * sizeof(*buffer));

    #pragma omp for schedule(static)
    for(idx_t x=0; x < bootstrap->nnz; ++x) {
      vals[x] = tc_predict_val(model, bootstrap, x, buffer);
    }

    splatt_free(buffer);
  }


  tc_model_free(model);

  return bootstrap;
}



/******************************************************************************
 * SPLATT-GENERATE
 *****************************************************************************/
int splatt_gen_cmd(
  int argc,
  char ** argv)
{
  gen_cmd_args args;
  default_gen_opts(&args);
  argp_parse(&gen_argp, argc, argv, ARGP_IN_ORDER, 0, &args);
  print_header();

  srand(args.seed);

  /* allocate model + workspace */
  omp_set_num_threads(args.nthreads);

  printf("GENERATING ------------------------------------------------------\n");
  printf("NFACTORS=%"SPLATT_PF_IDX" ", args.nfactors);
  if(args.set_seed) {
    printf("SEED=%u ", args.seed);
  } else {
    printf("SEED=time ");
  }
  printf("THREADS=%"SPLATT_PF_IDX"\n", args.nthreads);

  sptensor_t * ttrand = NULL;

  switch(args.which_alg) {
  case SPLATT_GEN_BOOTSTRAP:
    printf("ALG=BOOTSTRAP\n\n");
    ttrand = p_rand_bootstrap(&args);
    break;

  default:
    /* error */
    fprintf(stderr, "\n\nSPLATT: unknown generation algorithm\n");
    return SPLATT_ERROR_BADINPUT;
  }

  /* write the tensor */
  tt_write(ttrand, args.ofname);

  /* cleanup */
  tt_free(ttrand);

  return EXIT_SUCCESS;
}


