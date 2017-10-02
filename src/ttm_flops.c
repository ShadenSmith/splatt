
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "ttm.h"
#include "util.h"




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Compute the number of requisite flops for nodes for modee 'mode' when
*        performing TTMc at level 'ttmc_mode'. This function handles all mode
*        permutation, so 'mode' and 'ttmc_mode' should be absolute, not CSF.
*
* @param mode The mode to compute the cost.
* @param ttmc_mode The mode we are performing TTMc on.
* @param csf The CSF tensor to analyze.
* @param nfactors The rank of each mode.
*
* @return The number of requisite flops per node.
*/
static size_t p_mode_flops(
    idx_t const mode,
    idx_t const ttmc_mode,
    idx_t const nmodes,
    splatt_csf const * const csf,
    idx_t const * const nfactors)
{
  size_t flops = 0;

  idx_t const * const dim_perm = csf->dim_perm;

  /* permute nfactors */
  idx_t ranks[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    ranks[m] = nfactors[dim_perm[m]];
  }

  /* which level are we actually at in the tree? */
  idx_t const tree_level = csf_mode_to_depth(csf, mode);
  idx_t const outp_level = csf_mode_to_depth(csf, ttmc_mode);

  /*
   * Now account for three cases, whether we are above, below, or at the output
   * level in the tree.
   */

  /*
   * Output size. We can skip the first mode and just accumulate directly
   * into the output during the second level.
   */
  if(tree_level == outp_level && tree_level > 0) {
    idx_t ncols = 1;
    for(idx_t m=0; m < nmodes; ++m) {
      if(m != outp_level) {
        ncols *= ranks[m];
      }
    }
    flops = (size_t) ncols * 2;
  }

  /*
   * Push KPs down the tree. The first level requires no flops.
   */
  if(tree_level < outp_level) {
    /* grow KP size */
    idx_t ncols = 1;
    for(idx_t m=0; m <= tree_level; ++m) {
      ncols *= ranks[m];
    }
    if(tree_level > 0) {
      /* no 2 needed due to assignment, not accumulation */
      flops = (size_t) ncols;
    }
  }

  /*
   * Pull KPs up the tree.
   */
  if(tree_level > outp_level) {
    idx_t ncols = 1;
    for(idx_t m=tree_level; m < nmodes; ++m) {
      ncols *= ranks[m];
    }
    flops = (size_t) ncols * 2;
  }

  return flops;
}



/**
* @brief Count the requisite flops for performing TTMc on mode 'mode' using a
*        given CSF.
*
* @param csf The tensor to use.
* @param mode The mode to perform TTMc.
* @param nfactors The ranks of the modes.
*
* @return The requisite number of flops.
*/
static size_t p_count_csf_flops(
    splatt_csf const * const csf,
    idx_t const mode,
    idx_t const * const nfactors)
{
  idx_t const nmodes = csf->nmodes;
  idx_t const * const dim_perm = csf->dim_perm;

  size_t flops = 0;

  /* foreach tile */
  for(idx_t tile=0; tile < csf->ntiles; ++tile) {
    /* Sum the cost for each mode */
    for(idx_t m=0; m < csf->nmodes; ++m) {
      idx_t const csf_level = csf_mode_to_depth(csf, m);
      size_t const num_nodes = (size_t) csf->pt[tile].nfibs[csf_level];

      size_t node_flops = p_mode_flops(m, mode, nmodes, csf, nfactors);

      flops += node_flops * num_nodes;
    }
  }

  return flops;
}




/**
* @brief Count the flops required to perform mode-'mode' TTMc with a coordinate
*        tensor.
*
* @param tt The tensor.
* @param mode The mode we are interested in.
* @param nfactors The ranks of the modes.
*
* @return The number of flops.
*/
static size_t p_coord_count_flops(
    sptensor_t const * const tt,
    idx_t const mode,
    idx_t const * const nfactors)
{
  /* flops to grow kronecker products */
  size_t accum_flops = 0;
  size_t ncols = nfactors[0];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(m != mode) {
      accum_flops += (size_t) tt->nnz * ncols;
      ncols *= (size_t) nfactors[m];
    }
  }


  /* size of output */
  idx_t out_ncols = 1;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(m != mode) {
      out_ncols *= nfactors[m];
    }
  }

  /* add actual addition to output tensor */
  accum_flops += 2 * (size_t) tt->nnz * (size_t) out_ncols;

  return accum_flops;
}






/**
* @brief Traverse ALL permutations of tt->nmodes and fill a flop table.
*
* @param tt The tensor.
* @param nfactors The ranks of the modes.
* @param table The table to fill, must be at least (nmodes! x nmodes).
* @param perms The actual permutations.
* @param nperms running count or # permutations.
* @param prefix Permutation state.
* @param nprefix Permutation state.
* @param suffix Permutation state.
* @param nsuffix Permutation state.
*/
static void permute_all_csf(
    sptensor_t * const tt,
    idx_t const * const nfactors,
    size_t * const table,
    idx_t * const perms,
    idx_t * nperms,
    idx_t * prefix,
    idx_t nprefix,
    idx_t * suffix,
    idx_t nsuffix)
{
  /* if we have arrived at a permutation */
  if(nsuffix == 0) {
    double * opts = splatt_default_opts();
    opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

    /* Construct CSF */
    splatt_csf csf;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      csf.dim_perm[m] = prefix[m];

      idx_t const row = *nperms;
      perms[m + (row * tt->nmodes)] = prefix[m];
      printf(" %lu", prefix[m]);
    }

    for(idx_t m=0; m < tt->nmodes; ++m) {
      idx_t const row = *nperms;
    }
    printf("  ->");

    csf_alloc_mode(tt, CSF_MODE_CUSTOM, 0, &csf, opts);

    /* fill table row */
    size_t local_flops = 0;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      size_t const flops = p_count_csf_flops(&csf, m, nfactors);
      table[m + ((*nperms) * tt->nmodes)] = flops;
      printf("  %0.10e", (double) flops);
      local_flops += flops;
    }
    printf("  = %0.10e\n", (double) local_flops);

    /* next row */
    (*nperms)++;

    csf_free_mode(&csf);
    splatt_free_opts(opts);
    return;
  }


  /* continue with permutation generation */
  idx_t * new_prefix = splatt_malloc((nprefix+1) * sizeof(*new_prefix));
  idx_t * new_suffix = splatt_malloc((nsuffix-1) * sizeof(*new_prefix));

  for(idx_t i=0; i < nsuffix; ++i) {
    /* add suffix[j] to new_prefix */
    for(idx_t j=0; j < nprefix; ++j) {
      new_prefix[j] = prefix[j];
    }
    new_prefix[nprefix] = suffix[i];

    /* remove suffix[j] from suffix */
    idx_t ptr = 0;
    for(idx_t j=0; j < nsuffix; ++j) {
      if(j != i) {
        new_suffix[ptr++] = suffix[j];
      }
    }

    permute_all_csf(tt, nfactors, table, perms,
        nperms, new_prefix, nprefix + 1, new_suffix, nsuffix-1);
  }

  splatt_free(new_prefix);
  splatt_free(new_suffix);
}



static size_t p_ttmc_csf_optimal_flops(
    sptensor_t * const tt,
    idx_t const * const nfactors)
{
  idx_t const nmodes = tt->nmodes;
  /* nmodes! */
  idx_t nreps = 1;
  for(idx_t m=1; m <= nmodes; ++m) {
    nreps *= m;
  }

  size_t * table = splatt_malloc(nreps * nmodes * sizeof(*table));
  idx_t * perms  = splatt_malloc(nreps * nmodes * sizeof(*perms));

  /* now go over all permutations */
  idx_t * initial_perm = splatt_malloc(nmodes * sizeof(*initial_perm));
  for(idx_t m=0; m < nmodes; ++m) {
    initial_perm[m] = m;
  }

  idx_t nperms = 0;

  /* do all of the permutations */
  permute_all_csf(tt, nfactors, table, perms,
      &nperms, NULL, 0, initial_perm, nmodes);

  idx_t best_rep[MAX_NMODES];
  size_t best_flops[MAX_NMODES];

  /* now select optimal value for each mode */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    /* initialize with first representation */
    best_rep[m] = 0;
    best_flops[m] = table[m];

    /* foreach row */
    for(idx_t i=0; i < nreps; ++i) {
      size_t flops = table[m + (i * tt->nmodes)];

      if(flops < best_flops[m]) {
        best_flops[m] = flops;
        best_rep[m] = i;
      }
    }
  }

  size_t total_flops = 0;
  printf("CSF-O:");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    total_flops += best_flops[m];
    printf("  %0.10e", (double) best_flops[m]);
  }
  printf("   = %0.10e\n", (double) total_flops);
  printf("  OPTIMAL REPS:");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("  (");
    idx_t const row = best_rep[m];
    for(idx_t j=0; j < tt->nmodes; ++j) {
      printf("%"SPLATT_PF_IDX, perms[j + (row * tt->nmodes)]);
      if(j < tt->nmodes-1) {
        printf(", ");
      }
    }
    printf(")");
  }
  printf("\n");

  splatt_free(initial_perm);
  splatt_free(table);
  splatt_free(perms);
  return 0;
}




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void ttmc_fill_flop_tbl(
    sptensor_t * const tt,
    idx_t const * const nfactors,
    size_t table[MAX_NMODES][MAX_NMODES])
{
  /* just assume no tiling... */
  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

  /* flops if we used just CSF-1 or CSF-A */
  size_t csf1[MAX_NMODES];
  size_t csf2[MAX_NMODES];
  size_t csfa[MAX_NMODES];

  idx_t const smallest_mode = argmin_elem(tt->dims, tt->nmodes);
  idx_t const largest_mode = argmax_elem(tt->dims, tt->nmodes);

  /* foreach CSF rep */
  for(idx_t i=0; i < tt->nmodes; ++i) {
    printf("MODE-%"SPLATT_PF_IDX":  ", i);

    splatt_csf csf;
    csf_alloc_mode(tt, CSF_SORTED_SMALLFIRST_MINUSONE, i, &csf, opts);

    /* foreach mode of computation */
    for(idx_t j=0; j < tt->nmodes; ++j) {

      size_t const flops = p_count_csf_flops(&csf, j, nfactors);
      /* store result */
      table[i][j] = flops;
      printf("%0.10e  ", (double)flops);

      if(i == smallest_mode) {
        csf1[j] = flops;
        if(j != largest_mode) {
          csf2[j] = flops;
        }
      }
      if(i == j) {
        csfa[i] = flops;

        /* csf-2 uses special leaf mode */
        if(i == largest_mode) {
          csf2[j] = flops;
        }
      }
    } /* end foreach mode of computation */

    size_t total = 0;
    for(idx_t m=0; m < tt->nmodes; ++m) {
      total += table[i][m];
    }
    printf(" = %0.10e\n", (double)total);

    csf_free_mode(&csf);
  }
  splatt_free_opts(opts);
  printf("\n");



  /* print stats for each allocation scheme */

  size_t total;

  /* csf-1 and csf-a */
  total = 0;
  printf("CSF-1:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("%0.10e  ", (double)csf1[m]);
    total += csf1[m];
  }
  printf(" = %0.10e\n", (double)total);
  total = 0;
  printf("CSF-2:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("%0.10e  ", (double)csf2[m]);
    total += csf2[m];
  }
  printf(" = %0.10e\n", (double)total);

  total = 0;
  printf("CSF-A:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    printf("%0.10e  ", (double)csfa[m]);
    total += csfa[m];
  }
  printf(" = %0.10e\n", (double)total);

  bool mode_used[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mode_used[m] = false;
  }

  /* handpick best modes */
  printf("CUSTM:  ");
  total = 0;
  /* foreach mode */
  for(idx_t j=0; j < tt->nmodes; ++j) {
    size_t best = 0;
    /* foreach csf */
    for(idx_t i=0; i < tt->nmodes; ++i) {
      if(table[i][j] <= table[best][j]) {
        best = i;
      }
    }

    mode_used[best] = true;

    total += table[best][j];
    printf("%0.10e  ", (double) table[best][j]);
  }
  printf(" = %0.10e\n", (double) total);
  /* print CSF needed */
  printf("  CUSTOM MODES:");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(mode_used[m]) {
      printf(" %"SPLATT_PF_IDX, m);
    }
  }
  printf("\n");

#define COMPUTE_OPTIMAL_FLOPS
#ifdef  COMPUTE_OPTIMAL_FLOPS
  p_ttmc_csf_optimal_flops(tt, nfactors);
#endif

  printf("\n");
  /* coordinate form */
  total = 0;
  printf("COORD:  ");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    size_t const coord_flops = p_coord_count_flops(tt, m, nfactors);
    printf("%0.10e  ", (double)coord_flops);
    total += coord_flops;
  }
  printf(" = %0.10e\n", (double)total);
  printf("\n");

}



splatt_csf * ttmc_choose_csf(
    sptensor_t * const tt,
    idx_t const * const nfactors,
    idx_t max_tensors,
    idx_t * num_chosen_tensors,
    idx_t * csf_assignments)
{
  /* Bound number of CSF representations. */
  max_tensors = SS_MIN(max_tensors, tt->nmodes);

  /*
   * First construct the 'max_tensors' best tensors that we can.
   */

  /* Allocate the CSF pointers. */
  splatt_csf * * tensors = splatt_malloc(max_tensors * sizeof(*tensors));
  for(idx_t c=0; c < max_tensors; ++c) {
    tensors[c] = splatt_malloc(sizeof(**tensors));
  }

  /* Allocate table of flop costs. */
  size_t * table = splatt_malloc(max_tensors * tt->nmodes * sizeof(*table));

  /* Marker for modes which have already been optimized. */
  bool mode_optimized[MAX_NMODES];
  for(idx_t m=0; m < tt->nmodes; ++m) {
    mode_optimized[m] = false;
  }

  double * opts = splatt_default_opts();
  opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

  /* The first CSF is always CSF-1. */
  csf_alloc_mode(tt, CSF_SORTED_SMALLFIRST, 0, tensors[0], opts);
  for(idx_t m=0; m < tt->nmodes; ++m ) {
    table[m] = p_count_csf_flops(tensors[0], m, nfactors);
  }
  mode_optimized[tensors[0]->dim_perm[0]] = true;

  /* Now select the remaining (max_tensors - 1) modes greedily. */
  for(idx_t csf_rep=1; csf_rep < max_tensors; ++csf_rep) {
    idx_t worst_mode   = MAX_NMODES; /* invalid value */
    size_t worst_flops = 0;
    /* select the most expensive mode */
    for(idx_t c=0; c < csf_rep; ++c) {
      for(idx_t m=0; m < tt->nmodes; ++m) {
        /* don't construct the same CSF twice. */
        if(mode_optimized[m]) {
          continue;
        }

        size_t const flops = table[m + (c*tt->nmodes)];
        if(flops > worst_flops) {
          worst_flops = flops;
          worst_mode = m;
        }
      }
    } /* select most expensive mode */

    /* construct CSF and count flops */
    csf_alloc_mode(tt, CSF_SORTED_SMALLFIRST_MINUSONE, worst_mode,
        tensors[csf_rep], opts);
    for(idx_t m=0; m < tt->nmodes; ++m ) {
      size_t const flops = p_count_csf_flops(tensors[csf_rep], m, nfactors);
      table[m + (csf_rep * tt->nmodes)] = flops;
    }
    mode_optimized[worst_mode] = true;
  } /* greedy CSF construction */



  printf("GREEDY:");
  size_t total_flops = 0;

  /*
   * Now go over CSF's and select the best for each mode.
   */
  bool * csf_used = splatt_malloc(max_tensors * sizeof(*csf_used));
  for(idx_t c=0; c < max_tensors; ++c) {
    csf_used[c] = false;
  }
  *num_chosen_tensors = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t best_rep = 0;
    size_t best_flops = table[m];
    for(idx_t c=0; c < max_tensors; ++c) {
      size_t const flops = table[m + (c * tt->nmodes)];
      if(flops < best_flops) {
        best_flops = flops;
        best_rep = c;
      }
    }
    csf_assignments[m] = best_rep;
    if(!csf_used[best_rep]) {
      ++(*num_chosen_tensors);
    }
    csf_used[best_rep] = true;

    total_flops += best_flops;
    printf("  %0.10e", (double) best_flops);
  }
  printf("  = %0.10e\n", (double) total_flops);


  /*
   * Remove unused CSFs and compress csf_assignments[] to account for new
   * numbering.
   */
  splatt_csf * csf_ret = splatt_malloc((*num_chosen_tensors) * sizeof(*csf_ret));
  idx_t csf_ptr = 0;
  for(idx_t c=0; c < max_tensors; ++c) {
    if(csf_used[c]) {
      /* copy over structure contents -- shallow copy is correct here */
      memcpy(&(csf_ret[csf_ptr]), tensors[c], sizeof(*csf_ret));

      /* update numbering in assignments */
      for(idx_t m=0; m < tt->nmodes; ++m) {
        if(csf_assignments[m] == c) {
          csf_assignments[m] = csf_ptr;
        }
      }
      ++csf_ptr;

    } else {
      csf_free_mode(tensors[c]);
    }

    /* just delete the pointer */
    splatt_free(tensors[c]);
  }

  splatt_free_opts(opts);
  splatt_free(table);
  splatt_free(csf_used);
  splatt_free(tensors);

  return csf_ret;
}






