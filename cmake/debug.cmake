
set(SPLATT_WARNINGS false)

# check for debug mode
if (DEFINED DEBUG)
  message("Building in DEBUG mode. Assertions enabled.")
  add_definitions(-DDEBUG=${DEBUG})
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

  set(SPLATT_WARNINGS true)
else()
  message("Building in RELEASE mode.")
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
endif()

# Development mode enables much more strict warnings
if(DEFINED DEV_MODE)
  add_definitions(-DDEV_MODE=${DEV_MODE})
  set(SPLATT_WARNINGS true)
else()
endif()

include(cmake/warnings.cmake)

