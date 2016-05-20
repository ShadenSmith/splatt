/**
* @file global_options.h
* @brief Structures and function for global SPLATT options.
* @author Shaden Smith
* @version 2.0.0
* @date 2016-05-14
*/

#ifndef SPLATT_API_GLOBAL_OPTIONS_H
#define SPLATT_API_GLOBAL_OPTIONS_H


/******************************************************************************
 * TYPES
 *****************************************************************************/

typedef struct
{
  int num_threads;
  splatt_verbosity_type verbosity;

  int random_seed;
} splatt_global_opts;







/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif


/**
* @brief Allocate a global options structure and initialize with sane defaults.
*
* @return A pointer to the options, to be freed by `splatt_free_global_opts()`.
*/
splatt_global_opts * splatt_alloc_global_opts(void);


/**
* @brief Free a structure of global options.
*
* @param opts The structure to free.
*/
void splatt_free_global_opts(
    splatt_global_opts * opts);


#ifdef __cplusplus
}
#endif

#endif
