function [dim] = splatt_dim(X, varargin)
% SPLATT-DIM  Return the dimensions of a SPLATT CSF tensor.
%
% [dims] = splatt_dim(X);
% [dim] = splatt_dim(X, 1);
%
% If given no additional arguments, SPLATT-DIM returns a list of the tensor
% dimensions. You can optionally supply an argument to receive a single mode.
%
% See also splatt_cpd, splatt_mttkrp

if nargin == 1
  dim = X{1}.dims;
else
  dim = X{1}.dims(varargin{1});
end
