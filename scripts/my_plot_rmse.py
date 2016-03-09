#!/usr/bin/env python3

import sys
import re

# plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if len(sys.argv) == 1:
  print('usage: {} <run files>'.format(sys.argv[0]))
  sys.exit(1)

alg_re = re.compile('ALG=(\w+)')
epoch_re = re.compile('epoch:\s*(\d+)')
vrmse_re = re.compile('RMSE-vl: (\d+\.\d+e[+-]\d+)')
tr_time_re = re.compile('time-tr: (\d+\.\d+)s')
ts_time_re = re.compile('time-ts: (\d+\.\d+)s')

#plt.xlabel('Time (seconds)')
plt.xlabel('Epoch')
plt.ylabel('RMSE (validation)')

plots = []

markers = {
  'ALS' : '+-',
  'GD'  : 'x-',
  'SGD' : '.-',
  'CCD' : '^-',
  'LBFGS' : '*-',
}

# parse each file
for log in sys.argv[1:-1]:
  print(log)

  alg = None
  times = []
  rmses = []
  epochs = []

  with open(log, 'r') as logfile:
    for line in logfile:
      m = alg_re.search(line)
      if m:
        alg = m.group(1)

      m = epoch_re.search(line)
      if m:
        epoch = int(m.group(1))
        # grab RMSE and timing
        rmse = float(vrmse_re.search(line).group(1))
        time = float(tr_time_re.search(line).group(1)) + \
            float(ts_time_re.search(line).group(1))

        rmses.append(rmse)
        times.append(time)
        epochs.append(epoch)

  plt.plot(epochs, rmses, markers[alg], label=log, markersize=8)

plt.legend()
plt.xscale('log')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.7])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6))
plt.savefig(sys.argv[-1])

#plt.show()
#pp = PdfPages(sys.argv[-1])
#pp.savefig(plt.figure())
#pp.close()
