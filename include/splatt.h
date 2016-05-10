#ifndef SPLATT_SPLATT_H
#define SPLATT_SPLATT_H




/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <inttypes.h>
#include <float.h>

#ifdef SPLATT_USE_MPI
#include <mpi.h>
#endif


/******************************************************************************
 * VERSION
 *****************************************************************************/
#define SPLATT_VER_MAJOR     1
#define SPLATT_VER_MINOR     1
#define SPLATT_VER_SUBMINOR  1





/******************************************************************************
 * SPLATT MODULES
 *****************************************************************************/


/* types & structures */
#include "splatt/types.h"
#include "splatt/constants.h"
#include "splatt/structs.h"


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

#include "splatt/api_csf.h"
#include "splatt/api_factorization.h"
#include "splatt/api_kernels.h"
#include "splatt/api_kruskal.h"
#include "splatt/api_mpi.h"
#include "splatt/api_options.h"
#include "splatt/api_version.h"

#endif
