#ifndef SPLATTLAB_SHARED_H
#define SPLATTLAB_SHARED_H

#include <splatt.h>


/******************************************************************************
 * OPTION PARSING
 *****************************************************************************/
typedef struct splatt_option
{
  char * name;
  int has_arg;
  int opt_id;
} splattlab_option_t;

static splattlab_option_t option_names[] =
{
  {"tol", 1, SPLATT_OPTION_TOLERANCE},
  {"its", 1, SPLATT_OPTION_NITER},
  {"threads", 1, SPLATT_OPTION_NTHREADS},
  {"verbosity", 1, SPLATT_OPTION_VERBOSITY},
  {NULL, 0, 0}
};


static void __parse_opts(
    mxArray const * const opts,
    double * const cpd_opts)
{
  if(!mxIsStruct(opts)) {
    mexErrMsgTxt("SPLATT expects options array to be a structure.\n");
    return;
  }

  splattlab_option_t * head = option_names;
  for(; head->name != NULL; ++head) {
    mxArray * mopt = mxGetField(opts, 0, head->name);
    if(!mopt) {
      continue;
    }

    if(head->has_arg == 1) {
      cpd_opts[head->opt_id] = (double) mxGetScalar(mopt);
    } else {
      cpd_opts[head->opt_id] = 1;
    }
  }
}

#endif
