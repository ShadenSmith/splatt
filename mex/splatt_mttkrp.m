function splatt_mttkrp
% SPLATT-MTTKRP Multiply a tensor by matrices on all but one mode, called the
% matricized tensor times Khatri-Rao product (MTTKRP).
%
% [M] = splatt_mttkrp(X, mats, mode);
%
% SPLATT-MTTKRP accepts X, a tensor in CSF format and a cell array of
% matrices to multiply. The cell array must have as many entries an there are
% modes in X. It returns the resulting matrix after multiplying X
% by the all matrices but mats{mode}.
%
% Example usage:
%   X = splatt_load('mytensor.tns');
%   for m=1:X{1}.nmodes
%     mats{m} = splatt_mttkrp(X{m}, mats, m);
%   end
%
% options is a structure with the following fields:
%   'threads' : number of threads to use (default: #cores)
%
% The output and options commands are optional.
%
% See also splatt_cpd, splatt_load
