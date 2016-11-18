#!/bin/bash

# Regular tests

BUILDDIR="build/$(uname -s)-$(uname -m)"
./${BUILDDIR}/bin/test

