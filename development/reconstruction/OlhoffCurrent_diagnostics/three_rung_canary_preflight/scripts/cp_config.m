function [cfg, meta] = cp_config(nelx, nely)
%CP_CONFIG  THE canary configuration: the VALIDATED three-rung policy.
%
%   It is not re-typed here.  It is obtained by CALLING the builder that
%   produced the validated C320 result --
%
%       three_rung_promotion_validation_retry1/scripts/tr_config.m
%
%   -- which itself calls two_branch_controller_validation/scripts/cv_config.m
%   for the already-validated stage-exhaustion arm and appends the single
%   three-rung factor move.levels = [0.04 0.02 0.01].  Nothing in this study
%   introduces, retunes or re-spells a scientific field.  The ONLY departure
%   from tr_config is runtime.name, which olhoffcurrent_config_hash excludes,
%   so the canary config hash is identical to the validated policy's hash at
%   the same mesh.
%
%   WHY NOT olhoffcurrent_config?  Because production STILL delegates to the
%   unpromoted preset duOlhoffFrozenM4, i.e. the legacy four-rung / beta /
%   designChange policy.  That is exactly the defect that invalidated the
%   September 11 nine-mesh campaign.  This driver therefore applies the
%   validated overrides EXPLICITLY, and DEPLOYMENT_PREFLIGHT.md discloses that
%   the production preset is still unpromoted.
%
%   See also CP_PREFLIGHT, CP_RUN.

[cfg, trMeta] = tr_config(nelx, nely);

cfg = olh.config.resolve(trMeta.upstreamPreset, cfg.provenance.overrides{:}, ...
        'runtime.name', sprintf('CAN3_%dx%d', nelx, nely));
cfg.provenance.productionPreset = trMeta.preset;
cfg.provenance.implementation   = 'analysis/OlhoffCurrent';

meta = trMeta;
meta.arm     = 'CANARY3';
meta.label   = ['three-rung canary: frozen two-branch stage exhaustion, ' ...
                'E = A OR B, ladder [0.04 0.02 0.01]'];
meta.builder = 'three_rung_promotion_validation_retry1/scripts/tr_config.m';
end
