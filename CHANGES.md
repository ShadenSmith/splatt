
1.1.2
=====
* Added CREDITS.md
* Binary file IO now supported.
* Better handle missing input files.
* Chains-on-chains partition available
* CPD now accepts --seed parameter for reproducibility (works in MPI too)
* LAPACK now used


1.1.1
=====
* Updated README.md to include MPI instructions.
* splatt-check now outputs 1-indexed .map files.
* splatt-check now sums duplicate nonzeros.
* Memory allocations are now 64-byte aligned with `posix_memalign`. Memory
  allocation can be adjusted by altering `splatt_malloc`.
* Fixed extern "C" typo in include/splatt.h


1.1.0
=====
* Added support for fine-grained MPI decomposition.
* Point-to-point communication now uses paired Irecv and Isend.
* Fixed setup bugs for >3D medium-grained decompositions.


1.0.0
=====
* First stable release, includes API changes.
