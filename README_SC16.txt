
An Exploration of Optimization Algorithms for High Performance Tensor Completion
Shaden Smith, Jongsoo Park, and George Karypis
Supercomputing 2016

This repository contains the source code used for experiments involving the
distributed SGD algorithm. Source code for the ALS and CCD++ algorithms can be
found in the 'sc16' branch.

This branch requires one additional compilation step:

$ pushd src/SpMP ; make ; popd

This assumes you have the Intel compiler (icpc) available. You can check
src/SpMP/Makefile for additional options.

Next, follow compilation instructions in 'README.md'. All tensor completion
algorithms can be accessed via:

$ ./build/Linux-x86_64/bin/splatt complete

Add a '--help' flag for usage instructions.

