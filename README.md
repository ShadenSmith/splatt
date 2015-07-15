The Surprisingly ParalleL spArse Tensor Toolkit
===============================================

SPLATT is a library and C API for sparse tensor factorization. SPLATT supports
shared-memory parallelism with OpenMP and (soon) distributed-memory parallelism
with MPI.


Tensor Format
-------------
SPLATT expects tensors to be stored in 1-indexed coordinate format with
nonzeros separated by newlines. Each line of of the file has the coordinates of
the nonzero  followed by the value, all separated by spaces.  The following is
an example 2x2x3 tensor with 5 nonzeros:

    1 1 2 1.5
    1 2 2 2.5
    2 1 1 3.7
    1 2 3 0.5
    2 1 2 4.1


Building & Installing
---------------------
In short,

    $ ./configure && make

will build the SPLATT library and its executable. The executable will be found
in `build/<arch>/bin/`. You can also run

    $ ./configure --help

to see additional build options. To install,

    $ make install

will suffice. The installation prefix can be chosen by adding a
'--prefix=DIR' flag to configure.


Executable
----------
After building, an executable will found in the `build/` directory (or the
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

### Example 1

    $ splatt check mytensor.tns  --fix=fixed.tns

This runs `splatt-check` on 'mytensor.tns' and writes the fixed tensor to
'fixed.tns'. The `splatt-check` routine finds empty slices and duplicate
nonzero entries. Empty slices are indices in any mode which do not have any
nonzero entries associated with them. Some SPLATT routines (including CPD)
expect there to be no empty slices, so running `splatt-check` on a new tensor
is recommended.

### Example 2

    $ splatt cpd mytensor.tns -r 25 -t 4

This runs `splatt-cpd` on 'mytensor.tns' and finds a rank-25 CPD of the tensor.
Adding '-t 4' instructs SPLATT to use four OpenMP threads during the
computation. SPLATT will use all available CPU cores by default.  The matrix
factors are written to `modeN.mat` and lambda, the vector for scaling, is
written to `lambda.mat`.


C/C++ API
---------
SPLATT provides a C API which is callable from C and C++. Installation not only
installs the SPLATT executable, but also the static library `libsplatt.a` and
the header `splatt.h`. To use the C API, include `splatt.h` and link against
the SPLATT library. SPLATT also depends on the OpenMP and math libraries. The
following is an example compilation command:

    $ gcc mycode.c -lsplatt -lgomp -lm
    $ g++ mycode.cc -lsplatt -lgomp -lm

### IO
Unless otherwise noted, SPLATT expects tensors to be stored in the compressed
sparse fiber (CSF) format. SPLATT provides two functions for forming a tensor
in CSF:

* `splatt_csf_load` reads a tensor from a file
* `splatt_csf_convert` converts a tensor from coordinate format to CSF


### Computation
* `splatt_cpd` computes the CPD and returns a Kruskal tensor
* `splatt_default_opts` allocates and returns an options array with defaults


### Cleanup
All memory allocated by the SPLATT API should be freed by these functions:

* `splatt_free_csf` deallocates a list of CSF tensors
* `splatt_free_opts` deallocates a SPLATT options array
* `splatt_free_kruskal` deallocates a Kruskal tensor

### Example
The following is an example usage of the SPLATT API:

    #include <splatt.h>

    /* allocate default options */
    double * cpd_opts = splatt_default_opts();

    /* load the tensor from a file */
    int ret;
    splatt_idx_t nmodes;
    splatt_csf_t * tt;
    ret = splatt_csf_load("mytensor.tns", &nmodes, &tt, cpd_opts);

    /* do the factorization! */
    splatt_kruskal_t factored;
    ret = splatt_cpd(10, nmodes, tt, cpd_opts, &factored);

    /* do some processing */
    for(splatt_idx_t m = 0; m < nmodes; ++m) {
      /* access factored.lambda and factored.factors[m] */
    }

    /* cleanup */
    splatt_free_csf(nmodes, tt);
    splatt_free_kruskal(&factored);
    splatt_free_opts(cpd_opts);


Please see `splatt.h` for further documentation of SPLATT structures and call
signatures.


Octave/Matlab API
----------------------------
SPLATT also provides an API callable from Octave and Matlab that wraps the C
API. To compile the interface just enter the `mex/` directory from either
Octave or Matlab and call `make`.

    >> cd mex
    >> make

After compilation the MEX files will be found in the current directory. You can
now call those functions directly:

    >> KT = splatt_cpd('mytensor.tns', 25);

`splatt_cpd` returns a structure with three fields:

* `U` a cell array of the factor matrices
* `lambda` the factor column norms absorbed into a vector
* `fit` quality of the CPD defined by: 1 - (norm(residual) / norm(X))

SPLATT also supports explicitly storing tensors in CSF form to avoid IO times
during successive factorizations,

    >> X = splatt_load('mytensor.tns');
    >> K25 = splatt_cpd(X, 25);
    >> K50 = splatt_cpd(X, 50);

SPLATT accepts non-default parameters via structures:

    >> opts = struct('its', 100, 'tol', 1e-8);
    >> XT = splatt_cpd(X, 25, opts);


Licensing
---------
SPLATT is released under the MIT License. Please see the 'LICENSE' file for
details.
