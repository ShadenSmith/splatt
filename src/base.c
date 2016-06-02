
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"





/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

int splatt_version_major(void)
{
  return SPLATT_VER_MAJOR;
}

int splatt_version_minor(void)
{
  return SPLATT_VER_MINOR;
}

int splatt_version_subminor(void)
{
  return SPLATT_VER_SUBMINOR;
}






/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void * splatt_malloc(
    size_t const bytes)
{
  void * ptr;
  posix_memalign(&ptr, 64, bytes);
  if(ptr == NULL) {
    fprintf(stderr, "SPLATT: posix_memalign() returned NULL.\n");
  }
  return ptr;
}


void splatt_free(
    void * ptr)
{
  free(ptr);
}


