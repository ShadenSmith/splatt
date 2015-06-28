#!/bin/bash

mkoctfile --mex splatt_cpd.c -lsplatt -lgomp
CFLAGS=-std=c99 mkoctfile --mex splatt_load.c -lsplatt -lgomp

