
# check for MT-Metis support
if (DEFINED USE_MTMETIS)
  message("Building with MT-Metis support.")
  add_definitions(-DSPLATT_USE_MTMETIS=${USE_MTMETIS})
  set(SPLATT_LIBS ${SPLATT_LIBS} mtmetis)
endif()

# check for PatoH support
if (DEFINED USE_PATOH)
  message("Building with PaToH support.")
  add_definitions(-DSPLATT_USE_PATOH=${USE_PATOH})
  set(SPLATT_LIBS ${SPLATT_LIBS} patoh)
endif()

# check for Ashado support
if (DEFINED USE_ASHADO)
  message("Building with Ashado support.")
  add_definitions(-DSPLATT_USE_ASHADO=${USE_ASHADO})
  set(SPLATT_LIBS ${SPLATT_LIBS} ashado)
endif()

