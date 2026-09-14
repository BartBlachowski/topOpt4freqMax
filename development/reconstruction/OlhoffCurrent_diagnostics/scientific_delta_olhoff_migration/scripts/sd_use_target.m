function [P, guard] = sd_use_target()
%SD_USE_TARGET  Put ONLY the production OlhoffCurrent implementation on the path,
%   through its own fail-closed gate (olhoffcurrent_paths), read-only.
P = sd_paths();
restoredefaultpath;
addpath(fullfile(P.audit, 'scripts'));
addpath(P.oc);
guard = olhoffcurrent_paths();
sd_assert_resolution(P.impl, {'olhoffSolve','innerLoop','mmasub','subsolv','genGrad', ...
    'assemble2D','eigSolve','applyFilter','prepFilter','deltaLambda','massScale','model2D'});
end
