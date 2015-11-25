#ifndef SPLATT_SVD_H
#define SPLATT_SVD_H

#include "base.h"

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

#define left_singulars splatt_left_singulars
EXTERNC void left_singulars(
    val_t * inmat,
    val_t * outmat,
    idx_t const nrows,
    idx_t const ncols,
    idx_t const rank);

#undef EXTERNC

#endif
