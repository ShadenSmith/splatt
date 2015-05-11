The Surprisingly ParalleL spArse Tensor Toolkit
===============================================

SPLATT is a library and C API for sparse tensor factorization. We support
shared-memory parallelism with OpenMP and (soon) distributed-memory parallelism
with MPI.


Building & Installing
---------------------
In short,

    $ ./configure && make

will build the SPLATT library and its executable. You can add a '--help' flag
to configure to see additional build options. To install,

    $ make install

will suffice. The installation prefix can be chosen by adding a
'--prefix=<DIR>' flag to configure.


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
to be no empty slices, so running splatt-check
on any new tensors is recommended.

**Example 2**

    $ splatt cpd mytensor.tns -r 25 -t 4

This runs splatt-cpd on 'mytensor.tns' and finds a rank-25 CPD of the tensor.
Adding '-t 4' instructs SPLATT to use four CPU threads during the computation.
The matrix factors are written to 'mode<N>.mat' and lambda, the vector for
scaling, is written to 'lambda.mat'. See the '--help' option to see available
output formats.


API
---
Coming soon!


Licensing
---------
SPLATT is released under the MIT License. Please see the 'LICENSE' file for
details.
