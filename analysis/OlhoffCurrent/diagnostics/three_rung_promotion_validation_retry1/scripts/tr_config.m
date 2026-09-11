function [cfg, meta, cfg4] = tr_config(nelx, nely)
%TR_CONFIG  The THREE-RUNG candidate configuration.
%
%   It is EXACTLY the already-validated two_branch_controller_validation 'C'
%   arm -- the frozen two-branch stage-exhaustion controller, E = A OR B, the
%   arm that produced the C320 oracle -- plus ONE field:
%
%       move.levels : [0.04 0.02 0.01 0.005]  ->  [0.04 0.02 0.01]
%
%   The four-rung arm is not re-typed here.  It is obtained by CALLING
%   cv_config, and its exact override list is then REPLAYED through
%   olh.config.resolve with the single new override appended, so the two
%   configurations cannot drift apart by editing.  tr_singlefactor proves the
%   result mechanically by diffing every schema leaf.
%
%   runtime.name is also set, and is the ONLY other difference.  The schema
%   declares it 'free-text run label; never read by solver mathematics' and
%   olhoffcurrent_config_hash excludes it, so the config hashes of the two arms
%   differ by move.levels alone.
%
%   Returns the four-rung arm as cfg4 so callers need not rebuild it.

LEVELS = [0.04 0.02 0.01];  % THE single new factor

[cfg4, cvMeta] = cv_config('C', nelx, nely);
ov = cfg4.provenance.overrides;          % the oracle arm's exact override list

cfg = olh.config.resolve(cvMeta.upstreamPreset, ov{:}, ...
        'move.levels',  LEVELS, ...
        'runtime.name', sprintf('TR3_C_%dx%d', nelx, nely));
cfg.provenance.productionPreset = cvMeta.preset;
cfg.provenance.implementation   = 'analysis/OlhoffCurrent';

meta = struct('arm','C3', ...
    'label','three-rung candidate: frozen two-branch stage exhaustion, E = A OR B, ladder [0.04 0.02 0.01]', ...
    'nelx', nelx, 'nely', nely, ...
    'preset', cvMeta.preset, 'upstreamPreset', cvMeta.upstreamPreset, ...
    'baseArm', 'two_branch_controller_validation cv_config(''C'')', ...
    'singleFactor', 'move.levels', 'levels', LEVELS, ...
    'maxOuter', cvMeta.maxOuter);
end
