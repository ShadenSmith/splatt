
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
* @param dim_perm The permutation of the modes.
* @param nfactors The rank of each mode.
*
* @return The number of requisite flops per node.
*/
static size_t p_mode_flops(
    idx_t const mode,
    idx_t const ttmc_mode,
    idx_t const nmodes,
    idx_t const * const dim_perm,
    idx_t const * const nfactors)
{
  size_t flops = 0;

  /* permute nfactors */
  idx_t ranks[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    ranks[m] = nfactors[dim_perm[m]];
  }

  /* which level are we actually at in the tree? */
  idx_t const tree_level = csf_mode_depth(mode, dim_perm, nmodes);
  idx_t const outp_level = csf_mode_depth(ttmc_mode, dim_perm, nmodes);

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
      idx_t const csf_level = csf_mode_depth(m, dim_perm, nmodes);
      size_t const num_nodes = (size_t) csf->pt[tile].nfibs[csf_level];

      size_t node_flops = p_mode_flops(m, mode, nmodes, dim_perm, nfactors);

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

  /* print CSF needed */
  printf("CUSTOM MODES:");
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(mode_used[m]) {
      printf(" %"SPLATT_PF_IDX, m);
    }
  }
  printf("\n");

}
