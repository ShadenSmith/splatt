function [inner] = splatt_innerprod(mttkrp, mat, lambda)
% SPLATT-INNERPROD  Return the inner product of a tensor and a Kruskal tensor.
%
% [inner] = splatt_innerprod(Unew, U{nmodes}, lambda);
%
% This operation relies on a previously-computed MTTKRP, the remaining matrix
% factor, and lambda. The inner product can be computed along any of the tensor
% modes and only requires that the MTTKRP was performed on the mode that 'mat'
% represents.
%
% NOTE: no CSF tensor is used as input, as the tensor values are already
% absorbed in the MTTKRP result!
%
% The inner product is commonly done at the end of an iteration of some
% factorization routine. We recommend caching the MTTKRP output ('Unew')
% without modifications (e.g., column normalization) and supplying that to this
% function.
%
% SPLATT-MTTKRP can be used if no cached output is available (at the cost of
% re-computing MTTKRP):
%
% [inner] = splatt_innerprod(splatt_mttkrp(X,U,1), U{1}, lambda);
%
% See also splatt_mttkrp

inner = sum(mttkrp .* mat, 1) * lambda;
