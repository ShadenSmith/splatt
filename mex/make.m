function make
c = computer;
switch c
case 'MACI64'
    mex splatt_load.c -I../include -L../build/Darwin-x86_64/lib ...
    -lsplatt -lgomp -lm
    mex splatt_cpd.c -I../include -L../build/Darwin-x86_64/lib ...
    -lsplatt -lgomp -lm

case 'GLNXA64'
  mex splatt_load.c -I../include -L../build/Linux-x86_64/lib ...
      -lsplatt -lgomp -lm
  mex splatt_cpd.c -I../include -L../build/Linux-x86_64/lib ...
      -lsplatt -lgomp -lm

case 'GLNX32'
    mex splatt_load.c -I../include -L../build/Linux-x86/lib ...
        -lsplatt -lgomp -lm
    mex splatt_cpd.c -I../include -L../build/Linux-x86/lib ...
        -lsplatt -lgomp -lm
    %mex -O -largeArrayDims
end
