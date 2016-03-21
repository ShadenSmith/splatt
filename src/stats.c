
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "stats.h"
#include "sptensor.h"
#include "ftensor.h"
#include "csf.h"
#include "io.h"
#include "reorder.h"
#include "util.h"

#include <limits.h>
#include <math.h>

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Output basic statistics about tt to STDOUT.
*
* @param tt The tensor to inspect.
* @param ifname The filename of tt. Can be NULL.
*/
static void p_stats_basic(
  sptensor_t const * const tt,
  char const * const ifname)
{
  printf("Tensor information ---------------------------------------------\n");
  printf("FILE=%s\n", ifname);
  printf("DIMS=%"SPLATT_PF_IDX, tt->dims[0]);
  for(idx_t m=1; m < tt->nmodes; ++m) {
    printf("x%"SPLATT_PF_IDX, tt->dims[m]);
  }
  printf(" NNZ=%"SPLATT_PF_IDX, tt->nnz);
  printf(" DENSITY=%e\n" , tt_density(tt));

  char * bytestr = bytes_str(tt->nnz * ((sizeof(idx_t) * tt->nmodes) + sizeof(val_t)));
  printf("COORD-STORAGE=%s\n", bytestr);
  printf("\n");
  free(bytestr);
}


/**
* @brief Compute statistics about a hypergraph partitioning of tt.
*
* @param tt The tensor to inspect.
* @param mode The mode to operate on.
* @param pfname The file containing the partitioning.
*/
static void p_stats_hparts(
  sptensor_t * const tt,
  idx_t const mode,
  char const * const pfname)
{
  if(pfname == NULL) {
    fprintf(stderr, "SPLATT ERROR: analysis type requires partition file\n");
    exit(1);
  }

  ftensor_t ft;
  ften_alloc(&ft, tt, mode, 0);
  idx_t const nvtxs = ft.nfibs;
  idx_t nhedges = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    nhedges += tt->dims[m];
  }


  idx_t nparts = 0;
  idx_t * parts = part_read(pfname, nvtxs, &nparts);
  idx_t * pptr = NULL;
  idx_t * plookup = NULL;
  build_pptr(parts, nparts, nvtxs, &pptr, &plookup);

  /* get stats on partition sizes */
  idx_t minp = tt->nnz;
  idx_t maxp = 0;
  for(idx_t p=0; p < nparts; ++p) {
    idx_t nnz = 0;
    for(idx_t f=pptr[p]; f < pptr[p+1]; ++f) {
      idx_t const findex = plookup[f];
      nnz += ft.fptr[findex+1] - ft.fptr[findex];
    }
    if(nnz < minp) {
      minp = nnz;
    }
    if(nnz > maxp) {
      maxp = nnz;
    }
  }

  printf("Partition information ------------------------------------------\n");
  printf("FILE=%s\n", pfname);
  printf("NVTXS=%"SPLATT_PF_IDX" NHEDGES=%"SPLATT_PF_IDX"\n", nvtxs, nhedges);
  printf("NPARTS=%"SPLATT_PF_IDX" LIGHTEST=%"SPLATT_PF_IDX" HEAVIEST="
         "%"SPLATT_PF_IDX" AVG=%0.1f\n",
    nparts, minp, maxp, (val_t)(ft.nnz) / (val_t) nparts);
  printf("\n");

  idx_t * unique[MAX_NMODES];
  idx_t nunique[MAX_NMODES];
  for(idx_t m=0; m < ft.nmodes; ++m) {
    unique[m] = (idx_t *) splatt_malloc(ft.dims[ft.dim_perm[m]]
      * sizeof(idx_t));
  }

  /* now track unique ind info for each partition */
  for(idx_t p=0; p < nparts; ++p) {
    for(idx_t m=0; m < ft.nmodes; ++m) {
      memset(unique[m], 0, ft.dims[ft.dim_perm[m]] * sizeof(idx_t));
      nunique[m] = 0;
    }

    idx_t nnz = 0;
    idx_t ptr = 0;
    for(idx_t f=pptr[p]; f < pptr[p+1]; ++f) {
      idx_t const findex = plookup[f];
      nnz += ft.fptr[findex+1] - ft.fptr[findex];

      /* find slice of findex */
      while(ft.sptr[ptr] < findex && ft.sptr[ptr+1] < findex) {
        ++ptr;
      }
      if(unique[0][ptr] == 0) {
        ++nunique[0];
        unique[0][ptr] = 1;
      }

      /* mark unique fids */
      if(unique[1][ft.fids[f]] == 0) {
        ++nunique[1];
        unique[1][ft.fids[f]] = 1;
      }

      for(idx_t j=ft.fptr[findex]; j < ft.fptr[findex+1]; ++j) {
        idx_t const jind = ft.inds[j];
        /* mark unique inds */
        if(unique[2][jind] == 0) {
          ++nunique[2];
          unique[2][jind] = 1;
        }
      }
    }

    printf("%"SPLATT_PF_IDX"  ", p);
    printf("fibs: %"SPLATT_PF_IDX"(%4.1f%%)  ", pptr[p+1] - pptr[p],
      100. * (val_t)(pptr[p+1]-pptr[p]) / nvtxs);
    printf("nnz: %"SPLATT_PF_IDX" (%4.1f%%)  ", nnz, 100. * (val_t)nnz / (val_t) tt->nnz);
    printf("I: %"SPLATT_PF_IDX" (%4.1f%%)  ", nunique[0],
      100. * (val_t)nunique[0] / (val_t) ft.dims[mode]);
    printf("K: %"SPLATT_PF_IDX" (%4.1f%%)  ", nunique[1],
      100. * (val_t)nunique[1] / (val_t) ft.dims[ft.dim_perm[1]]);
    printf("J: %"SPLATT_PF_IDX" (%4.1f%%)\n", nunique[2],
      100. * (val_t)nunique[2] / (val_t) ft.dims[ft.dim_perm[2]]);
  }


  for(idx_t m=0; m < ft.nmodes; ++m) {
    free(unique[m]);
  }
  free(parts);
  free(plookup);
  free(pptr);
  ften_free(&ft);
}

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void stats_tt(
  sptensor_t * const tt,
  char const * const ifname,
  splatt_stats_type const type,
  idx_t const mode,
  char const * const pfile)
{
  switch(type) {
  case STATS_BASIC:
    p_stats_basic(tt, ifname);
    break;
  case STATS_HPARTS:
    p_stats_hparts(tt, mode, pfile);
    break;
  default:
    fprintf(stderr, "SPLATT ERROR: analysis type not implemented\n");
    exit(1);
  }
}


void stats_csf(
  splatt_csf const *ct, int ncopies)
{
  printf("nmodes: %"SPLATT_PF_IDX" nnz: %"SPLATT_PF_IDX"\n", ct->nmodes,
      ct->nnz);
  printf("dims: %"SPLATT_PF_IDX"", ct->dims[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("x%"SPLATT_PF_IDX"", ct->dims[m]);
  }
  for (idx_t k=0; k < ncopies; ++k, ++ct) {
    printf(" (%"SPLATT_PF_IDX"", ct->dim_perm[0]);
    for(idx_t m=1; m < ct->nmodes; ++m) {
      printf("->%"SPLATT_PF_IDX"", ct->dim_perm[m]);
    }
    printf(")\n");
    /*printf("ntiles: %"SPLATT_PF_IDX" tile dims: %"SPLATT_PF_IDX"", ct->ntiles,
        ct->tile_dims[0]);
    for(idx_t m=1; m < ct->nmodes; ++m) {
      printf("x%"SPLATT_PF_IDX"", ct->tile_dims[m]);
    }

    idx_t empty = 0;
    for(idx_t t=0; t < ct->ntiles; ++t) {
      if(ct->pt[t].vals == NULL) {
        ++empty;
      }
    }

    printf("  empty: %"SPLATT_PF_IDX" (%0.1f%%)\n", empty,
        100. * (double)empty/ (double)ct->ntiles);*/

    unsigned long long total_width = 0;
    int max_width = 0;

    idx_t const * const restrict sptr = ct->pt[0].fptr[0];
    idx_t const * const restrict fptr = ct->pt[0].fptr[1];

    idx_t const * const restrict sids = ct->pt[0].fids[0];
    idx_t const * const restrict fids = ct->pt[0].fids[1];
    idx_t const * const restrict inds = ct->pt[0].fids[2];

    idx_t const nslices = ct->pt[0].nfibs[0];

    idx_t nfibers = sptr[nslices];
    printf("  nslices: %"SPLATT_PF_IDX" nfibers: %"SPLATT_PF_IDX"\n", nslices, nfibers);

    idx_t min_slice_nnz = INT_MAX;
    idx_t max_slice_nnz = 0;
    double sq_sum_slice_nnz = 0;

    idx_t min_fiber_nnz = INT_MAX;
    idx_t max_fiber_nnz = 0;
    double sq_sum_fiber_nnz = 0;

#ifdef SPLATT_WRITE_NNZ_HIST
    FILE *fp_slice, *fp_fiber;
    if (k == 0) {
      fp_slice = fopen("slice0.hist", "w");
      fp_fiber = fopen("fiber0.hist", "w");
    }
    else {
      fp_slice = fopen("slice1.hist", "w");
      fp_fiber = fopen("fiber1.hist", "w");
    }
#endif

#pragma omp parallel for reduction(min:min_slice_nnz,min_fiber_nnz) reduction(max:max_slice_nnz,max_fiber_nnz,max_width) reduction(+:sq_sum_slice_nnz,sq_sum_fiber_nnz,total_width)
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      idx_t slice_nnz = fptr[sptr[s+1]] - fptr[sptr[s]];
      min_slice_nnz = SS_MIN(min_slice_nnz, slice_nnz);
      max_slice_nnz = SS_MAX(max_slice_nnz, slice_nnz);
      sq_sum_slice_nnz += slice_nnz*slice_nnz;

#ifdef SPLATT_WRITE_NNZ_HIST
      fprintf(fp_slice, "%ld\n", slice_nnz);
#endif

      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        idx_t fiber_nnz = fptr[f+1] - fptr[f];
        min_fiber_nnz = SS_MIN(min_fiber_nnz, fiber_nnz);
        max_fiber_nnz = SS_MAX(max_fiber_nnz, fiber_nnz);
        sq_sum_fiber_nnz += fiber_nnz*fiber_nnz;

        idx_t min_ind = INT_MAX, max_ind = 0;
        for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
          min_ind = SS_MIN(inds[jj], min_ind);
          max_ind = SS_MAX(inds[jj], max_ind);
        }
        if (fptr[f+1] > fptr[f]) {
          int width = max_ind - min_ind;
          total_width += width;
          max_width = SS_MAX(width, max_width);
        }

#ifdef SPLATT_WRITE_NNZ_HIST
        fprintf(fp_fiber, "%ld\n", fiber_nnz);
#endif
      }
    }

    double avg_width = (float)total_width/nfibers;
    printf("  avg_width = %f max_width = %d\n", avg_width, max_width);

#ifdef SPLATT_WRITE_NNZ_HIST
    fclose(fp_slice);
    fclose(fp_fiber);
#endif

    double avg_slice_nnz = (double)ct->nnz/nslices;
    double stddev_slice_nnz = sqrt(sq_sum_slice_nnz/nslices - avg_slice_nnz*avg_slice_nnz);

    double avg_fiber_nnz = (double)ct->nnz/nfibers;
    double stddev_fiber_nnz = sqrt(sq_sum_fiber_nnz/nfibers - avg_fiber_nnz*avg_fiber_nnz);

    printf("  slice_nnz (min,max,avg,stddev): %ld %ld %f %f\n", min_slice_nnz, max_slice_nnz, avg_slice_nnz, stddev_slice_nnz);
    printf("  fiber_nnz (min,max,avg,stddev): %ld %ld %f %f\n", min_fiber_nnz, max_fiber_nnz, avg_fiber_nnz, stddev_fiber_nnz);
  }
}


void cpd_stats(
  splatt_csf const * const csf,
  idx_t const nfactors,
  double const * const opts)
{
  /* find total storage */
  size_t fbytes = csf_storage(csf, opts);
  size_t mbytes = 0;
  for(idx_t m=0; m < csf[0].nmodes; ++m) {
    mbytes += csf[0].dims[m] * nfactors * sizeof(val_t);
  }

  /* header */
  printf("Factoring "
         "------------------------------------------------------\n");
  printf("NFACTORS=%"SPLATT_PF_IDX" MAXITS=%"SPLATT_PF_IDX" TOL=%0.1e ",
      nfactors,
      (idx_t) opts[SPLATT_OPTION_NITER],
      opts[SPLATT_OPTION_TOLERANCE]);
  printf("THREADS=%"SPLATT_PF_IDX" ", (idx_t) opts[SPLATT_OPTION_NTHREADS]);

  printf("\n");

  /* CSF allocation */
  printf("CSF-ALLOC=");
  splatt_csf_type which_csf = (splatt_csf_type)opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which_csf) {
  case SPLATT_CSF_ONEMODE:
    printf("ONEMODE");
    break;
  case SPLATT_CSF_TWOMODE:
    printf("TWOMODE");
    break;
  case SPLATT_CSF_ALLMODE:
    printf("ALLMODE");
    break;
  }
  printf(" ");

  /* tiling info */
  printf("TILE=");
  splatt_tile_type which_tile = (splatt_tile_type)opts[SPLATT_OPTION_TILE];
  switch(which_tile) {
  case SPLATT_NOTILE:
    printf("NO");
    break;
  case SPLATT_DENSETILE:
    printf("DENSE TILE-DEPTH=%"SPLATT_PF_IDX,
        (idx_t)opts[SPLATT_OPTION_TILEDEPTH]);
    break;
  case SPLATT_CCPTILE:
    printf("CCP TILE-DEPTH=%"SPLATT_PF_IDX,
        (idx_t)opts[SPLATT_OPTION_TILEDEPTH]);
    break;
  case SPLATT_SYNCTILE:
    printf("SYNC");
    break;
  case SPLATT_COOPTILE:
    printf("COOP");
    break;
  }
  printf("\n");

  char * fstorage = bytes_str(fbytes);
  char * mstorage = bytes_str(mbytes);
  printf("CSF-STORAGE=%s FACTOR-STORAGE=%s", fstorage, mstorage);
  free(fstorage);
  free(mstorage);
  printf("\n\n");
}


#ifdef SPLATT_USE_MPI
void mpi_cpd_stats(
  splatt_csf const * const csf,
  idx_t const nfactors,
  double const * const opts,
  rank_info * const rinfo)
{
  /* find total storage */
  unsigned long fbytes = csf_storage(csf, opts);
  unsigned long mbytes = 0;
  for(idx_t m=0; m < csf[0].nmodes; ++m) {
    mbytes += csf[0].dims[m] * nfactors * sizeof(val_t);
  }
  /* get storage across all nodes */
  if(rinfo->rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &fbytes, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
        rinfo->comm_3d);
    MPI_Reduce(MPI_IN_PLACE, &mbytes, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
        rinfo->comm_3d);
  } else {
    MPI_Reduce(&fbytes, NULL, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, rinfo->comm_3d);
    MPI_Reduce(&mbytes, NULL, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, rinfo->comm_3d);
  }

  /* only master rank prints from here */
  if(rinfo->rank != 0) {
    return;
  }

  /* header */
  printf("Factoring "
         "------------------------------------------------------\n");
  printf("NFACTORS=%"SPLATT_PF_IDX" MAXITS=%"SPLATT_PF_IDX" TOL=%0.1e ",
      nfactors,
      (idx_t) opts[SPLATT_OPTION_NITER],
      opts[SPLATT_OPTION_TOLERANCE]);
  printf("RANKS=%d THREADS=%"SPLATT_PF_IDX" ", rinfo->npes,
      (idx_t) opts[SPLATT_OPTION_NTHREADS]);

  printf("\n");

  /* CSF allocation */
  printf("CSF-ALLOC=");
  splatt_csf_type which_csf = (splatt_csf_type)opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which_csf) {
  case SPLATT_CSF_ONEMODE:
    printf("ONEMODE");
    break;
  case SPLATT_CSF_TWOMODE:
    printf("TWOMODE");
    break;
  case SPLATT_CSF_ALLMODE:
    printf("ALLMODE");
    break;
  }
  printf(" ");

  /* tiling info */
  printf("TILE=");
  splatt_tile_type which_tile = (splatt_tile_type)opts[SPLATT_OPTION_TILE];
  switch(which_tile) {
  case SPLATT_NOTILE:
    printf("NO");
    break;
  case SPLATT_DENSETILE:
    printf("DENSE TILE-DEPTH=%"SPLATT_PF_IDX,
        (idx_t)opts[SPLATT_OPTION_TILEDEPTH]);
    break;
  case SPLATT_CCPTILE:
    printf("CCP TILE-DEPTH=%"SPLATT_PF_IDX,
        (idx_t)opts[SPLATT_OPTION_TILEDEPTH]);
    break;
  case SPLATT_SYNCTILE:
    printf("SYNC");
    break;
  case SPLATT_COOPTILE:
    printf("COOP");
    break;
  }
  printf("\n");

  char * fstorage = bytes_str(fbytes);
  char * mstorage = bytes_str(mbytes);
  printf("CSF-STORAGE=%s FACTOR-STORAGE=%s", fstorage, mstorage);
  free(fstorage);
  free(mstorage);
  printf("\n\n");
}



void mpi_global_stats(
  sptensor_t * const tt,
  rank_info * const rinfo,
  char const * const ifname)
{
  idx_t * tmpdims = tt->dims;
  idx_t tmpnnz = tt->nnz;
  tt->dims = rinfo->global_dims;
  tt->nnz = rinfo->global_nnz;

  /* print stats */
  stats_tt(tt, ifname, STATS_BASIC, 0, NULL);

  /* restore local stats */
  tt->dims = tmpdims;
  tt->nnz = tmpnnz;
}

void mpi_rank_stats(
  sptensor_t const * const tt,
  rank_info const * const rinfo)
{
  idx_t totnnz = 0;
  idx_t maxnnz = 0;
  idx_t totvolume = 0;
  idx_t maxvolume = 0;
  idx_t volume = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    /* if a layer has > 1 rank there is a necessary reduction step too */
    if(rinfo->decomp != SPLATT_DECOMP_COARSE && rinfo->layer_size[m] > 1) {
      volume += 2 * (rinfo->nlocal2nbr[m] + rinfo->nnbr2globs[m]);
    } else {
      volume += rinfo->nlocal2nbr[m] + rinfo->nnbr2globs[m];
    }
  }
  MPI_Reduce(&volume, &totvolume, 1, SPLATT_MPI_IDX, MPI_SUM, 0, rinfo->comm_3d);
  MPI_Reduce(&volume, &maxvolume, 1, SPLATT_MPI_IDX, MPI_MAX, 0, rinfo->comm_3d);
  MPI_Reduce(&tt->nnz, &totnnz, 1, SPLATT_MPI_IDX, MPI_SUM, 0, rinfo->comm_3d);
  MPI_Reduce(&tt->nnz, &maxnnz, 1, SPLATT_MPI_IDX, MPI_MAX, 0, rinfo->comm_3d);

  if(rinfo->rank == 0) {
    printf("MPI information ------------------------------------------------\n");
    switch(rinfo->decomp) {
    case SPLATT_DECOMP_COARSE:
      printf("DISTRIBUTION=COARSE\n");
      break;
    case SPLATT_DECOMP_MEDIUM:
      printf("DISTRIBUTION=MEDIUM ");
      printf("DIMS=%d", rinfo->dims_3d[0]);
      for(idx_t m=1; m < rinfo->nmodes; ++m) {
        printf("x%d", rinfo->dims_3d[m]);
      }
      printf("\n");
      break;
    case SPLATT_DECOMP_FINE:
      printf("DISTRIBUTION=FINE\n");
      break;
    }
    idx_t avgvolume = totvolume / rinfo->npes;

    idx_t const avgnnz = totnnz / rinfo->npes;
    double nnzimbalance = 100. * ((double)(maxnnz - avgnnz) / (double)maxnnz);
    double volimbalance = 100. * ((double)(maxvolume - avgvolume) /
        SS_MAX((double)maxvolume, 1));
    printf("AVG NNZ=%"SPLATT_PF_IDX"\nMAX NNZ=%"SPLATT_PF_IDX"  (%0.2f%% diff)\n",
        avgnnz, maxnnz, nnzimbalance);
    printf("AVG COMMUNICATION VOL=%"SPLATT_PF_IDX"\nMAX COMMUNICATION VOL=%"SPLATT_PF_IDX"  "
        "(%0.2f%% diff)\n", avgvolume, maxvolume, volimbalance);
    printf("\n");
  }
}
#endif


