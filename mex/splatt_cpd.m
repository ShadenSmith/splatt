function splatt_cpd
% SPLATT-CPD  Call splatt-cpd from Matlab.
%
% [K] = splatt_cpd(X,nfactors) computes the rank-'nfactors' factorization of X
% using default options.
%
% [K] = splatt_cpd(X,nfactors,options) computes the rank-'nfactors' CPD of X
% using user-specified options.
%
% options is a structure with the following fields:
%   'tol'     : minimum fit-change for convergence (default: 1e-4)
%   'iters'   : maximum number of iterations to run (default: 50)
%   'verbose' : verbosity level from 0 to 2 (default: 1)
%   'threads' : number of threads to use (default: #cores)
%
% The output and options commands are optional.
%
% See also SPLATT-LOAD
