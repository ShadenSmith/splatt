/**
* @file api_version.h
* @brief Functions and macros for querying SPLATT version information.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/

#ifndef SPLATT_SPLATT_VERSION_H
#define SPLATT_SPLATT_VERSION_H




/******************************************************************************
 * MACROS
 *****************************************************************************/
#define SPLATT_VER_MAJOR     2
#define SPLATT_VER_MINOR     0
#define SPLATT_VER_SUBMINOR  0






/******************************************************************************
 * API - useful for shared libraries
 *****************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif

/**
\defgroup api_version_list List of functions for \splatt version numbering.
@{
*/

/**
* @brief Return the major version number of SPLATT. Matching major versions are
*        guaranteed to have backwards-compatible APIs.
*
* @return The major version number of SPLATT.
*/
int splatt_version_major(void);

/**
* @brief Return the minor version number of SPLATT.
*
* @return The minor version number of SPLATT.
*/
int splatt_version_minor(void);

/**
* @brief Return the subminor version number of SPLATT.
*
* @return The subminor version number of SPLATT.
*/
int splatt_version_subminor(void);


/** @} */


#ifdef __cplusplus
}
#endif

#endif
