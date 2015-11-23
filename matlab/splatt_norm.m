function [xnorm] = splatt_norm(X)
% SPLAT-NORM  Compute the Frobenius norm of a SPLATT CSF tensor.
%
% [xnorm] = splatt_norm(X);
%
% See also splatt_load, splatt_innerprod, splatt_mttkrp

xnorm = norm(X{1}.pt{:}.vals);
