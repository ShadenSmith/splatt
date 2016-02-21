#!/usr/bin/env python

import sys, re

matrix = ''
for line in open(sys.argv[1]):
  match = re.match('.*/(.*\.mtx).*', line)
  if match:
    matrix = match.group(1)

  match = re.match('fwd_p2p_tr_red_perm.*\s+(\S+) gbps', line)
  if match:
    PCL = match.group(1)

  match = re.match('fwd_mkl\s+.*\s+(\S+) gbps', line)
  if match:
    MKL2 = match.group(1)

  match = re.match('fwd_mkl_old.*\s+(\S+) gbps', line)
  if match:
    MKL = match.group(1)
    print matrix, MKL, MKL2, PCL
    matrix = None
    MKL = None
    MKL2 = None
    PCL = None
