
#
# OS specific stuff -- feel free to move to other cmake/ files if something
# fits better.
#

# Linux
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  # timing library
  set(SPLATT_LIBS ${SPLATT_LIBS} rt)
endif()

# OSX
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  find_library(argp_path argp)
  get_filename_component(argp_lib_dir ${argp_path} DIRECTORY)
  get_filename_component(argp_prefix ${argp_lib_dir} DIRECTORY)
  if("${argp_prefix}" STREQUAL "")
    message(WARNING "Can't find path to argp library; bravely continuing on")
  else()
    include_directories(${argp_prefix}/include/)
    link_directories(${argp_prefix}/lib/)
  endif()
  set(SPLATT_LIBS ${SPLATT_LIBS} argp)
endif()

