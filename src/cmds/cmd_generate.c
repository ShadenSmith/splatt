
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

static char gen_args_doc[] = "-i INPUT [OUTPUT]";
static char gen_doc[] =
  "splatt-generate -- Create a sparse tensor from a random CPD factorization.\n"
  "Available tensor generation are:\n"
  "  bootstrap\t\tuse the sparsity pattern of an existing tensor\n"
  "  split\t\t\tsplit a tensor into train/validate/test tensors\n";


#define GEN_TRAIN 254
#define GEN_SEED 255
static struct argp_option gen_options[] = {
  {"input", 'i', "TENSOR", 0, "sparse tensor to randomize"},
  {"rank", 'r', "RANK", 0, "rank (default: 10)"},
  {"type", 't', "TYPE", 0, "which algorithm to use (default: bootstrap)"},
  {"train", GEN_TRAIN, "FRACTION", 0, "size (as fraction of input) of training set (default: 0.80)"},
  {"freq", 'f', "COUNT", 0, "keep at least this many appearances of a slice in the training set (default: 3)"},
  {"seed", GEN_SEED, "SEED", 0, "random seed (default: system time)"},
  {0}
};


typedef enum
{
  SPLATT_GEN_RAND,
  SPLATT_GEN_BOOTSTRAP,
  SPLATT_GEN_SPLIT,
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
  { "split", SPLATT_GEN_SPLIT },
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

  /* split options */
  double train_frac;
  idx_t freq;

  idx_t nfactors;
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
  args->set_seed = false;
  args->seed = time(NULL);
  args->train_frac = 0.80;
  args->freq = 5;
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
    args->which_alg = parse_gen_alg(arg);
    if(args->which_alg == SPLATT_GEN_NALGS) {
      fprintf(stderr, "SPLATT: unknown generation algorithm '%s'.\n", arg);
      argp_usage(state);
    }
    break;
  case 'r':
    args->nfactors = strtoull(arg, &buf, 10);
    break;
  case 'f':
    args->freq = strtoull(arg, &buf, 10);
    break;
  case GEN_SEED:
    args->seed = (unsigned int) atoi(arg);
    args->set_seed = true;
    break;
  case GEN_TRAIN:
    args->train_frac = atof(arg);
    break;

  case ARGP_KEY_ARG:
    if(args->ofname != NULL) {
      argp_usage(state);
      break;
    }
    args->ofname = arg;
    break;

  case ARGP_KEY_END:
    break;
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
*/
static int p_rand_bootstrap(
    gen_cmd_args * args)
{
  if(args->ofname == NULL) {
    fprintf(stderr, "SPLATT: bootstrap generation requires input and output "
                    "files.\n");
    return SPLATT_ERROR_BADINPUT;
  }

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

  /* write the tensor */
  tt_write(bootstrap, args->ofname);

  /* cleanup */
  tt_free(bootstrap);

  return SPLATT_SUCCESS;
}



static int p_rand_split(
    gen_cmd_args * args)
{
  /* load input tensor */
  sptensor_t * input = tt_read(args->ifname);
  if(input == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }
  stats_tt(input, args->ifname, STATS_BASIC, 0, NULL);

  idx_t const nmodes = input->nmodes;
  idx_t const train_freq = args->freq;

  idx_t const train_tgt = args->train_frac * input->nnz;
  idx_t const test_tgt = (input->nnz - train_tgt) / 2;
  idx_t const val_tgt = input->nnz - (train_tgt + test_tgt);
  assert(train_tgt + test_tgt + val_tgt == input->nnz);

  printf("TRAIN-TARGET=%"SPLATT_PF_IDX" TEST-TARGET=%"SPLATT_PF_IDX" "
         "VALIDATION-TARGET=%"SPLATT_PF_IDX"\n", train_tgt, test_tgt, val_tgt);

  /* randomize nonzero ordering */
  idx_t * perm = splatt_malloc(input->nnz * sizeof(*perm));
  #pragma omp parallel for
  for(idx_t x=0; x < input->nnz; ++x) {
    perm[x] = x;
  }
  shuffle_idx(perm, input->nnz);

  /* get histograms of each slice */
  idx_t * hists[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    hists[m] = tt_get_hist(input, m);
  }

  /* Allocate train, test, and validation tensors. In the worst case, train
   * is every nonzero because no indices meet the minimum freq. */
  sptensor_t * train = tt_alloc(input->nnz, nmodes);
  sptensor_t * test  = tt_alloc(test_tgt, nmodes);
  sptensor_t * val   = tt_alloc(val_tgt, nmodes);
  /* initially 0 nonzeros */
  train->nnz = 0;
  test->nnz = 0;
  val->nnz = 0;

  idx_t train_ptr = 0;
  idx_t test_ptr = 0;
  idx_t val_ptr = 0;

  /* extract test set */
  for(idx_t nnz_ptr=0; nnz_ptr < input->nnz; ++nnz_ptr) {
    idx_t const idx = perm[nnz_ptr];

    /* check for validity */
    bool valid = true;
    for(idx_t m=0; m < nmodes; ++m) {
      if(hists[m][input->ind[m][idx]] <= train_freq) {
        valid = false;
        break;
      }
    }

    /* save to test */
    if(valid) {
      test->vals[test_ptr] = input->vals[idx];
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t const cpy_ind = input->ind[m][idx];
        test->ind[m][test_ptr] = cpy_ind;
        --(hists[m][cpy_ind]);
      }
      if(++test_ptr == test_tgt) {
        break;
      }

    /* save to training */
    } else {
      train->vals[train_ptr] = input->vals[idx];
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t const cpy_ind = input->ind[m][idx];
        train->ind[m][train_ptr] = cpy_ind;
      }
      ++train_ptr;
    }
  }


  /* extract validation set */
  for(idx_t nnz_ptr = train_ptr + test_ptr; nnz_ptr < input->nnz; ++nnz_ptr) {
    idx_t const idx = perm[nnz_ptr];

    /* check for validity */
    bool valid = true;
    for(idx_t m=0; m < nmodes; ++m) {
      if(hists[m][input->ind[m][idx]] <= train_freq) {
        valid = false;
        break;
      }
    }

    /* save to validation */
    if(valid) {
      val->vals[val_ptr] = input->vals[idx];
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t const cpy_ind = input->ind[m][idx];
        val->ind[m][val_ptr] = cpy_ind;
        --(hists[m][cpy_ind]);
      }
      if(++val_ptr == val_tgt) {
        break;
      }

    /* save to training */
    } else {
      train->vals[train_ptr] = input->vals[idx];
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t const cpy_ind = input->ind[m][idx];
        train->ind[m][train_ptr] = cpy_ind;
      }
      ++train_ptr;
    }
  }

  /* grab rest of input */
  idx_t const sofar = train_ptr + test_ptr + val_ptr;
  for(idx_t nnz_ptr = sofar; nnz_ptr < input->nnz; ++nnz_ptr) {
    idx_t const idx = perm[nnz_ptr];
    train->vals[train_ptr] = input->vals[idx];
    for(idx_t m=0; m < nmodes; ++m) {
      idx_t const cpy_ind = input->ind[m][idx];
      train->ind[m][train_ptr] = cpy_ind;
    }
    ++train_ptr;
  }

  /* save sizes */
  train->nnz = train_ptr;
  test->nnz = test_ptr;
  val->nnz = val_ptr;

  printf("TRAIN-NNZ=%"SPLATT_PF_IDX" TEST-NNZ=%"SPLATT_PF_IDX" "
         "VALIDATION-NNZ=%"SPLATT_PF_IDX"\n", train->nnz, test->nnz, val->nnz);
  assert(train->nnz + test->nnz + val->nnz == input->nnz);

  tt_write(train, "train.tns");
  tt_write(test, "test.tns");
  tt_write(val, "val.tns");

  /* cleanup */
  tt_free(train);
  tt_free(test);
  tt_free(val);
  splatt_free(perm);
  for(idx_t m=0; m < nmodes; ++m) {
    splatt_free(hists[m]);
  }

  return SPLATT_SUCCESS;
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
  printf("GENERATING -----------------------------------------------------\n");
  if(args.set_seed) {
    printf("SEED=%u ", args.seed);
  } else {
    printf("SEED=time ");
  }

  int success = SPLATT_SUCCESS;
  switch(args.which_alg) {
  case SPLATT_GEN_BOOTSTRAP:
    printf("ALG=BOOTSTRAP NFACTORS=%"SPLATT_PF_IDX"\n\n", args.nfactors);
    success =p_rand_bootstrap(&args);
    break;
  case SPLATT_GEN_SPLIT:
    printf("ALG=SPLIT TRAIN-FRAC=%0.2f MIN-FREQ=%"SPLATT_PF_IDX"\n\n",
        args.train_frac, args.freq);
    success = p_rand_split(&args);
    break;

  default:
    /* error */
    fprintf(stderr, "\n\nSPLATT: unknown generation algorithm\n");
    return SPLATT_ERROR_BADINPUT;
  }

  return success;
}


