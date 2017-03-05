
# default widths
set(CONFIG_VAL_WIDTH 64)
set(CONFIG_IDX_WIDTH 64)
set(CONFIG_BLAS_INT int32_t)

# check for user-defined widths
if (DEFINED USER_IDX_WIDTH)
  if (${USER_IDX_WIDTH} STREQUAL "32")
    set(CONFIG_IDX_WIDTH 32)
  elseif (${USER_IDX_WIDTH} STREQUAL "64")
    set(CONFIG_IDX_WIDTH 64)
  else()
    message(FATAL_ERROR "Width '${USER_IDX_WIDTH}' not recognized.\
      Choose between {32 64}.")
  endif()
  message("Using ${CONFIG_IDX_WIDTH}-bit integers.")
endif()

if (DEFINED USER_VAL_WIDTH)
  if (${USER_VAL_WIDTH} STREQUAL "single")
    set(CONFIG_VAL_WIDTH 32)
  elseif (${USER_VAL_WIDTH} STREQUAL "double")
    set(CONFIG_VAL_WIDTH 64)
  else()
    message(FATAL_ERROR "Precision '${USER_VAL_WIDTH}' not recognized.\
      Choose between {single, double}.")
  endif()
  message("Using ${CONFIG_VAL_WIDTH} precision floating point numbers.")
endif()


if (DEFINED USER_BLAS_INT)
  set(CONFIG_BLAS_INT ${USER_BLAS_INT})
  message("Using type '${CONFIG_BLAS_INT}' for BLAS/LAPACK integers.")
endif()

# Configure include/splatt/types.h to include specified type widths.
configure_file(${CMAKE_SOURCE_DIR}/include/splatt/types_config.h
               ${CMAKE_SOURCE_DIR}/include/splatt/types.h)

