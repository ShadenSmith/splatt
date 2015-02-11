#ifndef SPLATT_MPI_H
#define SPLATT_MPI_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"
#include <mpi.h>


/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

sptensor_t * mpi_tt_read(
  char const * const ifname);

void tt_distribute_stats(
  sptensor_t * const tt,
  idx_t const nprocs);

#endif
