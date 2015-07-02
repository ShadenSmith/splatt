The Surprisingly ParalleL spArse Tensor Toolkit
===============================================

SPLATT is a library and C API for sparse tensor factorization. We support
shared-memory parallelism with OpenMP and (soon) distributed-memory parallelism
with MPI.


Building & Installing
---------------------
In short,

    $ ./configure && make

will build the SPLATT library and its executable. The executable will be found
in build/<arch>/bin. You can add a '--help' flag to configure to see additional
build options. To install,

    $ make install

will suffice. The installation prefix can be chosen by adding a
'--prefix=DIR' flag to configure.


Executable
----------
After building, an executable will found in the build/ directory (or the
installation prefix if SPLATT was installed). SPLATT builds a single executable
which features a number of sub-commands:

* cpd
* check
* convert
* reorder
* stats

All SPLATT commands are executed in the form

    $ splatt CMD [OPTIONS]

You can execute

    $ splatt CMD --help

for usage information of each command.

**Example 1**

    $ splatt check mytensor.tns  --fix=fixed.tns

This runs splatt-check on 'mytensor.tns' and writes the fixed tensor to
'fixed.tns'. The splatt-check routine finds empty slices and duplicate nonzero
entries. Empty slices are indices in any mode which do not have any nonzero
entries associated with them. Some SPLATT routines (including CPD) expect there
to be no empty slices, so running splatt-check on any new tensors is
recommended.

**Example 2**

    $ splatt cpd mytensor.tns -r 25 -t 4

This runs splatt-cpd on 'mytensor.tns' and finds a rank-25 CPD of the tensor.
Adding '-t 4' instructs SPLATT to use four CPU threads during the computation.
The matrix factors are written to 'modeN.mat' and lambda, the vector for
scaling, is written to 'lambda.mat'. See the '--help' option to see available
output formats.


API
---
SPLATT provides a C API which is callable from C and C++. The `splatt.h` header
file must be included. Please see the documented header file for call
signatures.

**IO**

Unless otherwise noted, SPLATT expects tensors to be stored in the compressed
sparse fiber (CSF) format. SPLATT provides two functions for forming a tensor
in CSF:

* `splatt_csf_load` reads a tensor from a file
* `splatt_csf_convert` converts a tensor from coordinate format to CSF


**Computation**

* `splatt_cpd` computes the CPD and returns a Kruskal tensor
* `splatt_default_opts` allocates and returns an options array with defaults


**Cleanup**

All memory allocated by the SPLATT API should be freed by these functions:

* `splatt_free_csf` deallocates a list of CSF tensors
* `splatt_free_opts` deallocates a SPLATT options array
* `splatt_free_kruskal` deallocates a Kruskal tensor

**Example**

The following is an example usage of the SPLATT API:

    /* allocate default options */
    double * cpd_opts = splatt_default_opts();

    /* load the tensor from a file */
    splatt_idx_t nmodes;
    splatt_csf_t ** tt = splatt_csf_load("mytensor.tns", &nmodes, cpd_opts);

    /* do the factorization! */
    splatt_kruskal_t factored;
    splatt_cpd(10, nmodes, tt, cpd_opts, &factored);

    /* cleanup */
    splatt_csf_free(nmodes, tt);
    splatt_free_kruskal(&factored);
    splatt_free_opts(cpd_opts);


Licensing
---------
SPLATT is released under the MIT License. Please see the 'LICENSE' file for
details.
