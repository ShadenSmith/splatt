
project(CSPLATT)
cmake_minimum_required(VERSION 2.6.0)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git/)
  execute_process(COMMAND git --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git rev-parse --short HEAD
                  OUTPUT_VARIABLE SPLATT_COMMIT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND git --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git rev-parse --abbrev-ref HEAD
                  OUTPUT_VARIABLE SPLATT_BRANCH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(SPLATT_VERSION_STR "${SPLATT_COMMIT}-${SPLATT_BRANCH}")
  set(C_DEFINES "${C_DEFINES} -DVERSION_STR=\\\"${SPLATT_VERSION_STR}\\\"")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_DEFINES}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-variable")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftree-vectorize")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopt-info-vec")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopt-info-vec-missed")

file(GLOB SOURCES src/*.c)

# Generate splatt library
add_subdirectory(lib)

# Compile applications
add_subdirectory(apps)

# UNIT TESTS
#add_subdirectory(test)

# copy scripts directory
add_custom_target(
  ScriptsDir ALL
  DEPENDS scripts/Contents.txt
)

add_custom_command(
  COMMAND cp -R ${CMAKE_CURRENT_SOURCE_DIR}/scripts ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Copying scripts"
  OUTPUT scripts/Contents.txt
  VERBATIM
)

