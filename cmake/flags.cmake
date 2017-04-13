
set(CMAKE_C_STANDARD   99)
set(CMAKE_CXX_STANDARD 11)

set(SPLATT_FLAGS "${SPLATT_FLAGS} -funroll-loops")
set(SPLATT_FLAGS "${SPLATT_FLAGS} -fstrict-aliasing")
set(SPLATT_FLAGS "${SPLATT_FLAGS} -fPIC")

if(${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -xHost")
  set(SPLATT_FLAGS  "-restrict")
  add_definitions(-D_XOPEN_SOURCE=600)
else()
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -march=native")
  set(SPLATT_FLAGS "${SPLATT_FLAGS} -ftree-vectorize")
endif()

