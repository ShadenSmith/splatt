
# Get version string from git repo if possible
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git/)
  execute_process(COMMAND git --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git rev-parse --short HEAD
                  OUTPUT_VARIABLE SPLATT_COMMIT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND git --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git rev-parse --abbrev-ref HEAD
                  OUTPUT_VARIABLE SPLATT_BRANCH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(SPLATT_VERSION_STR "${SPLATT_COMMIT}-${SPLATT_BRANCH}")
  set(C_DEFINES "${C_DEFINES} -DSPLATT_VERSION_STR=\\\"${SPLATT_VERSION_STR}\\\"")
endif()
