
if (DEFINED USE_FORTRAN)
  set(SPLATT_NOWARN ${USE_FORTRAN})

  # Enable linking against Fortran
  enable_language(Fortran)

  # Link against a supplied Fortran library.
  if (DEFINED USER_FORTRAN_LIB)
    message(STATUS "Using user supplied Fortran library=${USER_FORTRAN_LIB}")
    set(SPLATT_LIBS ${SPLATT_LIBS} ${USER_FORTRAN_LIB})

  # Try popular ones.
  else()

    # TODO search better
    if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "Intel")
      set(SPLATT_LIBS ${SPLATT_LIBS} ifcore)
    else()
      set(SPLATT_LIBS ${SPLATT_LIBS} gfortran)
    endif()
  endif()
endif()

