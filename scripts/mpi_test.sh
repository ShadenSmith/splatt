#!/bin/bash

# MPI tests
mpirun -np 4 ./build/Linux-x86_64/bin/test_mpi
mpirun -np 7 ./build/Linux-x86_64/bin/test_mpi
