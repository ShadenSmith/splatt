function splatt_cpd
% SPLATT-CPD  Compute the Canonical Polyadic Decomposition (CPD) using SPLATT.
%
% [K] = splatt_cpd(filename, nfactors);
% [K] = splatt_cpd(X, nfactors);
% [K] = splatt_cpd(..., options);
%
% SPLATT-CPD accepts either a filename or a tensor in CSF format and returns
% its CPD. The 'nfactors' parameter specifies the rank of the decomposition.
% If read from a file, nonzero indices are expected to be 1-indexed. If
% SPLATT-CPD is to be called repeatedly (i.e., for exploration of various
% nfactors parameters), SPLATT-LOAD can be used to avoid repeated IO and
% pre-processing costs.
%
% options is a structure with the following fields:
%   'tol'     : minimum fit-change for convergence (default: 1e-4)
%   'iters'   : maximum number of iterations to run (default: 50)
%   'verbose' : verbosity level from 0 to 2 (default: 1)
%   'threads' : number of threads to use (default: #cores)
%
% The output and options commands are optional.
%
% See also splatt_load, splatt_mttkrp
