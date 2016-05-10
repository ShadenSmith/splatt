#ifndef SPLATT_SPLATT_VERSION_H
#define SPLATT_SPLATT_VERSION_H


/*
 * VERSION API
 */



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
