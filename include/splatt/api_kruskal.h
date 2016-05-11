/**
* @file api_kruskal.h
* @brief Functions for manipulating tensors stored in Kruskal form (after CPD).
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/



#ifndef SPLATT_SPLATT_KRUSKAL_H
#define SPLATT_SPLATT_KRUSKAL_H


/*
 * KRUSKAL TENSOR API
 */


#ifdef __cplusplus
extern "C" {
#endif

/**
\defgroup api_kruskal_list List of functions for \splatt Kruskal tensors.
@{
*/

/**
* @brief Free a splatt_kruskal allocated by splatt_cpd().
*
* @param factored The factored tensor to free.
*/
void splatt_free_kruskal(
    splatt_kruskal * factored);

/** @} */


#ifdef __cplusplus
}
#endif

#endif
