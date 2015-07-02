function make_octave
c = computer;
switch c
case 'x86_64-pc-linux-gnu'
  mkoctfile --mex splatt_load.c -I../include -L../build/Linux-x86_64/lib ...
      -lsplatt -lgomp -lm
  mkoctfile --mex  splatt_cpd.c -I../include -L../build/Linux-x86_64/lib ...
      -lsplatt -lgomp -lm

% TODO: What does it use for OSX?
case 'MACI64'
    mkoctfile --mex splatt_load.c -I../include -L../build/Darwin-x86_64/lib ...
    -lsplatt -lgomp -lm
    mkoctfile --mex splatt_cpd.c -I../include -L../build/Darwin-x86_64/lib ...
    -lsplatt -lgomp -lm

% TODO: What does it do for 32-bit Linux?
case 'GLNX32'
    mkoctfile --mex splatt_load.c -I../include -L../build/Linux-x86/lib ...
        -lsplatt -lgomp -lm
    mkoctfile --mex splatt_cpd.c -I../include -L../build/Linux-x86/lib ...
        -lsplatt -lgomp -lm
    %mex -O -largeArrayDims
end
