/**
* @file api_options.h
* @brief Functions for supplying optional parameters to the SPLATT API.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/



#ifndef SPLATT_SPLATT_OPTION_H
#define SPLATT_SPLATT_OPTION_H


/*
 * OPTIONS API
 */



#ifdef __cplusplus
extern "C" {
#endif


/**
\defgroup api_opt_list List of functions for \splatt options.
@{
*/


/**
* @brief Allocate and fill an options array with default options.
*
* @return The options array.
*/
double * splatt_default_opts(void);


/**
* @brief Free an options array allocated with splatt_default_opts().
*/
void  splatt_free_opts(
    double * opts);



/** @} */


#ifdef __cplusplus
}
#endif

#endif
