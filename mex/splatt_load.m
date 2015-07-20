function splatt_load
% SPLATT-LOAD  Load a tensor and convert to CSF format. Returns a cell array of
% the CSF tensors for each mode.
%
% [X] = splatt_load('filename');
% [X] = splatt_load(inds, vals);
% [X] = splatt_load(..., options);
%
% SPLATT-LOAD accepts a tensor input as either a filename or matrices of
% indices and values. The 'inds' matrix should be of dimension (nnz x nmodes)
% and 'vals' should be (nnz x 1). Indices in the file or matrix are expected to
% be 1-indexed. The matrices can be from Tensor Toolbox's 'sptensor' format and
% the calling sequence would be:
%
% [X] = splatt_load(mysptensor.subs, mysptensor.vals);
%
% options is a structure with the following fields:
%   'mode' : which mode to represent in CSF: from 1 to nmodes or 'all'
%            (default: 'all')
%   'tile' : type of cache tiling to use: 'none', 'sync', or 'coop'
%            (default: 'none')
%
% The output and options commands are optional.
%
% See also splatt_cpd, splatt_mttkrp
