
# type widths
if (DEFINED SPLATT_IDX_WIDTH)
  add_definitions(-DSPLATT_IDX_TYPEWIDTH=${SPLATT_IDX_WIDTH})
  message("Using ${SPLATT_IDX_WIDTH}-bit integers.")
endif()

if (DEFINED SPLATT_VAL_WIDTH)
  if (${SPLATT_VAL_WIDTH} STREQUAL "single")
    add_definitions(-DSPLATT_VAL_TYPEWIDTH=32)
  elseif (${SPLATT_VAL_WIDTH} STREQUAL "double")
    add_definitions(-DSPLATT_VAL_TYPEWIDTH=64)
  else()
    message(FATAL_ERROR "Precision '${SPLATT_VAL_WIDTH}' not recognized.\
      Choose between {single, double}.")
  endif()
  message("Using ${SPLATT_VAL_WIDTH} precision floating point numbers.")
endif()

