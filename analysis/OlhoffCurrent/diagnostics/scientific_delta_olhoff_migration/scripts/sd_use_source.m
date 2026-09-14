function P = sd_use_source()
%SD_USE_SOURCE  Put ONLY the immutable source snapshot on the MATLAB path.
%   Mirrors the snapshot's own setpaths.m (fem, filter, algo, mma, architecture)
%   after restoredefaultpath, keeps this scripts folder, and asserts that every
%   solver symbol used resolves inside the snapshot and nowhere else.
P = sd_paths();
restoredefaultpath;
addpath(fullfile(P.audit, 'scripts'));
s = P.snap;
addpath(fullfile(s,'fem'), fullfile(s,'filter'), fullfile(s,'algo'), fullfile(s,'mma'));
addpath(fullfile(s,'architecture'));
sd_assert_resolution(s, {'olhoffSolve','innerLoop','mmasub','subsolv','genGrad', ...
    'assemble2D','eigSolve','applyFilter','prepFilter','deltaLambda','massScale','model2D'});
end
