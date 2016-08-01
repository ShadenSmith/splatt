
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

if (DEFINED USE_METIS)
  add_definitions(-DSPLATT_USE_METIS=${USE_METIS})

  if (DEFINED USER_METIS_LIB)
  message("Using user supplied Metis=${USER_METIS_LIB}")
    set(SPLATT_LIBS ${SPLATT_LIBS} ${USER_METIS_LIB})

  # try to find it
  else()
    find_library(metis_path metis)
    get_filename_component(metis_lib_dir ${metis_path} DIRECTORY)
    get_filename_component(metis_prefix ${metis_lib_dir} DIRECTORY)
    if("${metis_prefix}" STREQUAL "")
      message(ERROR "Can't find path to Metis library. Try --with-metis-lib=")
    else()
      include_directories(${metis_prefix}/include/)
      link_directories(${metis_prefix}/lib/)

      message(STATUS "Found Metis lib: ${metis_path}")
    endif()
    set(SPLATT_LIBS ${SPLATT_LIBS} metis)
  endif()

  set(SPLATT_NOWARN ${USE_METIS})
endif()
