#!/usr/bin/env python2

import re
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

fin = None
if len(sys.argv) == 1:
  fin = sys.stdin
else:
  fin = open(sys.argv[1])

mode_re    = re.compile('(\w+) =+')
threads_re = re.compile('## THREADS (\d+)')
m1_re      = re.compile('M1  avg = (\d+\.\d+)s')
fname_re   = re.compile('fname: .+/([\w_\.]+)\s.*')

# parse input into data structure
# times =>  {'FTENSOR' : { 1:10.0, 2:5.0, 4:2.5 }}
times = defaultdict(dict)
seen_threads = {}
ttname = 'PLOT NAME'
mode = None
curr_th = None
for line in fin:
  m = fname_re.match(line)
  if m:
    ttname = m.group(1)

  m = mode_re.match(line)
  if m:
    mode = m.group(1)

  m = threads_re.match(line)
  if m:
    curr_th = int(m.group(1))
    seen_threads[curr_th] = True

  m = m1_re.match(line)
  if m:
    time = float(m.group(1))
    times[mode][curr_th] = time


markers = { 'FTENSOR_short':'x', 'FTENSOR_long':'^', 'GTENSOR':'o', 'STENSOR':'+' }
colors  = { 'FTENSOR_short':'g', 'FTENSOR_long':'c', 'GTENSOR':'b', 'STENSOR':'m' }

# prepare plot
max_threads = max(seen_threads.keys())

# plot times
fig = plt.subplot(2,2,1)
plt.xlabel('# threads')
plt.ylabel('time (s)')
for mode in times.keys():
  xs = []
  ys = []
  for t in sorted(times[mode].keys()):
    xs.append(t)
    ys.append(times[mode][t])
  plt.plot(xs, ys, '--', label=mode, color=colors[mode], marker=markers[mode],
      markersize=7)

plt.setp(fig, xticks=sorted(seen_threads.keys()))
#plt.legend(loc='upper right')

# plot speedups
fig = plt.subplot(2,2,2)
plt.xlabel('# threads')
plt.ylabel('relative speedup')
# plot ideal line
plt.plot(range(1,max_threads+1), range(1,max_threads+1), '-', label='ideal')
#plot speedups
for mode in times.keys():
  xs = []
  ys = []
  serial = times[mode][1]
  for t in sorted(times[mode].keys()):
    xs.append(t)
    ys.append(serial / times[mode][t])

  plt.plot(xs, ys, '--', label=mode, color=colors[mode], marker=markers[mode],
      markersize=7)

plt.setp(fig, xticks=sorted(seen_threads.keys()))
#plt.setp(fig, yticks=sorted(seen_threads.keys()))
plt.legend(bbox_to_anchor=(0.20, -0.50), loc=2, borderaxespad=0.)

# plot relative speedups
fig = plt.subplot(2,2,3)
plt.xlabel('# threads')
plt.ylabel('speedup over serial GTENSOR')
#plot speedups
serial = times['GTENSOR'][1]
for mode in times.keys():
  if mode == 'GTENSOR':
    continue
  xs = []
  ys = []
  for t in sorted(times[mode].keys()):
    xs.append(t)
    ys.append(serial / times[mode][t])

  plt.plot(xs, ys, '--', label=mode, color=colors[mode], marker=markers[mode],
      markersize=7)
#plt.setp(fig, xticks=sorted(seen_threads.keys()))

plt.suptitle(ttname, fontsize=20, weight='bold')
plt.savefig('out.pdf')
#plt.show()

