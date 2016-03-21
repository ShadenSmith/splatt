#!/usr/bin/env python3

import sys
import re

# plotting
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
  print('usage: {} <run files>'.format(sys.argv[0]))
  sys.exit(1)

alg_re = re.compile('ALG=(\w+)')
epoch_re = re.compile('epoch:\s*(\d+)')
trmse_re = re.compile('RMSE-tr: (\d+\.\d+e[+-]\d+)')
vrmse_re = re.compile('RMSE-vl: (\d+\.\d+e[+-]\d+)')
time_re = re.compile('time: (\d+\.\d+)s')

plt.xlabel('Time (s)')
plt.ylabel('RMSE (train)')

plots = []

markers = {
  'ALS'   : '+-',
  'GD'    : 'x-',
  'NLCG'  : 'p-',
  'SGD'   : '.-',
  'CCD'   : '^-',
}

# parse each file
for log in sys.argv[1:]:
  alg = None
  times = []
  rmses = []

  with open(log, 'r') as logfile:
    for line in logfile:
      m = alg_re.search(line)
      if m:
        alg = m.group(1)

      m = epoch_re.search(line)
      if m:
        epoch = int(m.group(1))
        # grab RMSE and timing
        trmse = float(trmse_re.search(line).group(1))
        vrmse = float(vrmse_re.search(line).group(1))
        time = float(time_re.search(line).group(1))

        rmses.append(trmse)
        times.append(time)

  plt.plot(times, rmses, markers[alg], label=log, markersize=8)

plt.legend()
plt.show()

