
#include "base.h"


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
