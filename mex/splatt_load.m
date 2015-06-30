function splatt_load
% SPLATT-LOAD  Load a tensor from a file and convert to CSF format.
%
% [X] = splatt_load('filename', options);
%
% options is a structure with the following fields:
%   'mode' : which mode to represent in CSF: from 1 to nmodes or 'all'
%            (default: 'all')
%   'tile' : type of cache tiling to use: 'none', 'sync', or 'coop'
%            (default: 'none')
%
% The output and options commands are optional.
%
% See also SPLATT-CPD
