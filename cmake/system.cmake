
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
  # argparse (homebrew/argp-standalone)
  set(SPLATT_LIBS ${SPLATT_LIBS} argp)
endif()

