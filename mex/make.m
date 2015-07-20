% Adapted from MetisMEX
function make
  % octave uses mkoctfile instead of mex
  if(exist('OCTAVE_VERSION', 'builtin') ~= 0)
    make_octave;
    return;
  end

  c = computer;
  switch c
  case 'MACI64'
      mex splatt_load.c -I../include -L../build/Darwin-x86_64/lib ...
      -lsplatt -lgomp -lm
      mex splatt_cpd.c -I../include -L../build/Darwin-x86_64/lib ...
      -lsplatt -lgomp -lm
      mex splatt_mttkrp.c -I../include -L../build/Darwin-x86_64/lib ...
      -lsplatt -lgomp -lm

  case 'GLNXA64'
    mex splatt_load.c -I../include -L../build/Linux-x86_64/lib ...
        -lsplatt -lgomp -lm
    mex splatt_cpd.c -I../include -L../build/Linux-x86_64/lib ...
        -lsplatt -lgomp -lm
    mex splatt_mttkrp.c -I../include -L../build/Linux-x86_64/lib ...
        -lsplatt -lgomp -lm

  case 'GLNX32'
      mex splatt_load.c -I../include -L../build/Linux-x86/lib ...
          -lsplatt -lgomp -lm
      mex splatt_cpd.c -I../include -L../build/Linux-x86/lib ...
          -lsplatt -lgomp -lm
      mex splatt_mttkrp.c -I../include -L../build/Linux-x86/lib ...
          -lsplatt -lgomp -lm
  end
