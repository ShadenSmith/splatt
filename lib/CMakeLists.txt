
project(SPLATT_LIB)
cmake_minimum_required(VERSION 2.8.0)

add_library(splatt ${SPLATT_SOURCES})

install(TARGETS splatt
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
