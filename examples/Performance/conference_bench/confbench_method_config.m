function [mcfg, profileId, profile] = confbench_method_config(methodKey, nelx, nely, outputDir)
%CONFBENCH_METHOD_CONFIG  The frozen scientific configuration of one method.
%
%   [mcfg, profileId, profile] = CONFBENCH_METHOD_CONFIG(methodKey, nelx, nely, outputDir)
%
%   The conference benchmark driver owns the RUN configuration -- which meshes,
%   which methods, where the output goes.  It does NOT own the science.  Each
%   method's scientific settings are read here from the artifact that froze
%   them, and nothing in this file selects, tunes or defaults a scientific
%   value:
%
%     Proposed  analysis/three_method_parametric_study/results/profile_freeze_manifest.json
%               profile proposed_practical_move02_tol001
%     Yuksel    the same manifest, profile yuksel_practical_move01_tol001
%     Olhoff    analysis/OlhoffCurrent, the SOLE production Du-Olhoff
%               implementation, resolved at its named production preset
%               duOlhoffFixedPenaltySensitivityFiltered
%
%   The Olhoff branch deliberately does NOT read
%   analysis/olhoff_stabilization_audit/final_campaign_profile.json.  That file
%   names the SUPERSEDED fixed-1600-iteration S1 profile; the conference
%   benchmark must not depend on it, even to read another method's entry.
%
%   It also does not read analysis/OlhoffM4Reconstruction, which is now frozen
%   historical evidence rather than production.  The production preset
%   reproduces that realization bitwise at 160x20 and 320x40 -- see
%   analysis/OLHOFF_CURRENT_PROMOTION_REPORT.md.
%
%   See also CONFBENCH_RUN_CASE, OLHOFFCURRENT_CONFIG, OLHOFFCURRENT_PRESET.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
study = fullfile(repo, 'analysis', 'three_method_parametric_study');
freezePath = fullfile(study, 'results', 'profile_freeze_manifest.json');

methodKey = lower(char(string(methodKey)));

switch methodKey
    case 'olhoff'
        % The production path guard must be installed before olh.config.* can
        % resolve: the canonical package lives inside +impl/ and is deliberately
        % invisible to genpath.  The guard is released when this function
        % returns; confbench_run_case installs its own for the solve.
        guard = olhoffcurrent_paths(); %#ok<NASGU>
        preset = olhoffcurrent_preset();
        cfg = olhoffcurrent_config(nelx, nely);

        % mcfg carries BOTH renderings.  The canonical cfg is the
        % configuration; the flat view exists only so that checks and manifests
        % written in the historical vocabulary keep working without anyone
        % re-deriving the realization by hand.
        mcfg = olhoffcurrent_legacy_view(cfg);
        mcfg.nelx = nelx;
        mcfg.nely = nely;
        mcfg.canonical = cfg;

        profileId = preset.name;
        profile = struct( ...
            'label',                preset.label, ...
            'production_preset',    preset.name, ...
            'upstream_preset',      preset.upstreamPreset, ...
            'classification',       preset.classification, ...
            'historical_aliases',   {preset.historicalAliases}, ...
            'must_not_be_labelled', preset.mustNotBeLabelled, ...
            'epistemic_class',      preset.epistemicClass, ...
            'caveat',               olhoffcurrent_caveat(), ...
            'source_implementation', ...
                'analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m', ...
            'frozen_by_file', 'analysis/OlhoffCurrent/olhoffcurrent_preset.m', ...
            'effective_config_hash', olhoffcurrent_config_hash(cfg));
        return

    case {'proposed', 'ourapproach'}
        freeze = jsondecode(fileread(freezePath));
        profile = freeze.profiles.proposed_practical;
        profileId = char(profile.profile_id);
        prm = struct('move', profile.move, 'rmin_element', profile.rmin_element, ...
            'max_iters', profile.max_iters, 'tol', profile.tol, ...
            'record_history', false);
        mcfg = study_base_config('proposed', nelx, nely, prm);

    case 'yuksel'
        freeze = jsondecode(fileread(freezePath));
        profile = freeze.profiles.yuksel_practical;
        profileId = char(profile.profile_id);
        prm = struct('move', profile.move, 'rmin_element', profile.rmin_element, ...
            'max_iters', profile.max_iters, 'tol', profile.stage2_tol, ...
            'stage1_tol', profile.stage1_tol, 'stage2_tol', profile.stage2_tol, ...
            'stage1_max_iters', profile.max_iters, 'record_history', false);
        mcfg = study_base_config('yuksel', nelx, nely, prm);

    otherwise
        error('confbench_method_config:UnknownMethod', ...
            'Unknown method key "%s".', methodKey);
end

% ---- dispatched methods only, from here down ---------------------------
mcfg.meta.profile_id = profileId;
mcfg.meta.frozen_by = 'analysis/three_method_parametric_study/results/profile_freeze_manifest.json';
mcfg.meta.source_implementation = char(profile.source_implementation);
mcfg.meta.threads_per_run = 1;
mcfg.postprocessing.visualize_live = false;
mcfg.postprocessing.save_final_image = false;
mcfg.postprocessing.save_snapshot_image = false;
if nargin >= 4 && ~isempty(outputDir)
    mcfg.meta.output_dir = char(outputDir);
end
end
