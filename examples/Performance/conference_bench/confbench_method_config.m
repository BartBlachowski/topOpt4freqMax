function [mcfg, profileId, profile] = confbench_method_config(methodKey, nelx, nely, outputDir, stop)
%CONFBENCH_METHOD_CONFIG  The frozen scientific configuration of one method.
%
%   [mcfg, profileId, profile] = CONFBENCH_METHOD_CONFIG(methodKey, nelx, nely, outputDir)
%   [...] = CONFBENCH_METHOD_CONFIG(..., stop)
%
%   stop (optional) changes this method's stopping rule away from the frozen
%   one; an absent or empty field keeps the frozen value.  It is the driver's
%   cfg.stop.<method> (performance_comparison.m), where the production values
%   are listed.  .criterion selects WHAT the tolerance is compared with, on the
%   design variable x (rho for Olhoff):
%     'max_change'          max|x - x_old|                 (Proposed/Yuksel native)
%     'relative_l2_change'  ||x - x_old||_2 / ||x_old||_2  (strict <, no
%                           production value: its tolerance must be given)
%     'l2_change'           Olhoff only, its native rule ||drho||_2 < c*sqrt(NE/3200)
%   Fields:
%     olhoff    .criterion 'l2_change' (default) | 'max_change' (max|drho| <=
%                          tol) | 'relative_l2_change'; the last two use no
%                          mesh scaling and no guards
%               .c         c in ||drho||_2 < c*sqrt(NE/3200)   (l2_change)
%               .tol       tolerance of max_change / relative_l2_change
%               .maxOuter  outer-iteration safety budget
%     proposed  .criterion 'max_change' (default) | 'relative_l2_change'
%               .tol       tolerance of the selected criterion
%               .maxIters  iteration safety budget
%     yuksel    .criterion 'max_change' (default) | 'relative_l2_change', both stages
%               .stage1Tol, .stage2Tol   per-stage tolerances of that criterion
%               .maxIters  per-stage safety budget, both stages (the driver
%                          passes cfg.yukselMaxIters)
%   The returned configuration is the effective one, so everything that
%   prints or hashes it reports the rule that actually runs.
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
%               implementation, resolved at the preset the benchmark names in
%               confbench_olhoff_preset (currently
%               duOlhoffPedersenAdaptiveBoxSensitivityFiltered, which the
%               preflight also requires to be the recorded production preset)
%
%   The Olhoff branch deliberately does NOT read
%   analysis/olhoff_stabilization_audit/final_campaign_profile.json.  That file
%   names the SUPERSEDED fixed-1600-iteration S1 profile; the conference
%   benchmark must not depend on it, even to read another method's entry.
%
%   It also does not read analysis/OlhoffM4Reconstruction, which is frozen
%   historical evidence.  That realization is reproduced bitwise by the
%   HISTORICAL preset duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered, which
%   is no longer the benchmark's Olhoff preset -- see
%   analysis/OlhoffCurrent/diagnostics/upstream_253069_migration.
%
%   See also CONFBENCH_RUN_CASE, OLHOFFCURRENT_CONFIG, OLHOFFCURRENT_PRESET.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
freezePath = fullfile(repo, 'examples', 'Performance', 'benchmark_profile', 'profile_freeze_manifest.json');

methodKey = lower(char(string(methodKey)));
if nargin < 5 || isempty(stop)
    stop = struct();
end
if isfield(stop, 'useC')
    error('confbench_method_config:UseCRetired', ...
        ['stop.useC was replaced by stop.criterion (''l2_change'' = useC true, ' ...
         '''max_change'' = useC false).']);
end

switch methodKey
    case 'olhoff'
        % The production path guard must be installed before olh.config.* can
        % resolve: the canonical package lives inside +impl/ and is deliberately
        % invisible to genpath.  The guard is released when this function
        % returns; confbench_run_case installs its own for the solve.
        guard = olhoffcurrent_paths(); %#ok<NASGU>
        preset = olhoffcurrent_preset(confbench_olhoff_preset());
        % l2_change reads .c; the other two criteria read .tol, which has no
        % production value for them.  The field a criterion does not read is
        % ignored.
        stopFactor = []; stopMaxTol = []; stopRelTol = [];
        criterion = char(valueOr(stop, 'criterion', 'l2_change'));
        switch criterion
            case 'l2_change'
                stopFactor = valueOr(stop, 'c', []);
            case {'max_change', 'relative_l2_change'}
                if ~hasValue(stop, 'tol')
                    error('confbench_method_config:OlhoffTolRequired', ...
                        'stop.olhoff.criterion = ''%s'' requires stop.olhoff.tol.', criterion);
                end
                if strcmp(criterion, 'max_change'); stopMaxTol = stop.tol;
                else;                               stopRelTol = stop.tol; end
            otherwise
                error('confbench_method_config:UnknownCriterion', ...
                    ['stop.olhoff.criterion must be ''l2_change'', ''max_change'' or ' ...
                     '''relative_l2_change'' (got ''%s'').'], criterion);
        end
        stopArgs = {};
        if ~isempty(stopFactor); stopArgs = [stopArgs, {'StopToleranceFactor', stopFactor}]; end
        if ~isempty(stopMaxTol); stopArgs = [stopArgs, {'StopMaxChangeTolerance', stopMaxTol}]; end
        if ~isempty(stopRelTol); stopArgs = [stopArgs, {'StopRelativeChangeTolerance', stopRelTol}]; end
        if hasValue(stop, 'maxOuter'); stopArgs = [stopArgs, {'MaxOuter', stop.maxOuter}]; end
        cfg = olhoffcurrent_config(nelx, nely, 'Preset', preset.name, stopArgs{:});

        % mcfg carries BOTH renderings.  The canonical cfg is the
        % configuration; the flat view exists only so that checks and manifests
        % written in the historical vocabulary keep working without anyone
        % re-deriving the realization by hand.
        mcfg = olhoffcurrent_legacy_view(cfg);
        mcfg.nelx = nelx;
        mcfg.nely = nely;
        mcfg.canonical = cfg;
        mcfg.olhoff_preset = preset.name;
        % Forwarded by confbench_run_case to olhoffcurrent_run, which resolves
        % the configuration again for the solve; empty = the preset's own.
        mcfg.olhoff_stop_factor = stopFactor;
        mcfg.olhoff_stop_max_change_tol = stopMaxTol;
        mcfg.olhoff_stop_relative_tol = stopRelTol;
        mcfg.olhoff_max_outer = valueOr(stop, 'maxOuter', []);

        profileId = preset.name;
        profile = struct( ...
            'label',                preset.label, ...
            'display_name',         preset.displayName, ...
            'production_preset',    preset.name, ...
            'preset_role',          preset.role, ...
            'upstream_preset',      preset.upstreamPreset, ...
            'upstream_commit',      preset.upstreamCommit, ...
            'classification',       preset.classification, ...
            'formulation',          preset.formulation, ...
            'compatibility_aliases', {preset.compatibilityAliases}, ...
            'historical_aliases',   {preset.historicalAliases}, ...
            'must_not_be_labelled', preset.mustNotBeLabelled, ...
            'epistemic_class',      preset.epistemicClass, ...
            'distinct_from',        preset.distinctFrom, ...
            'caveat',               olhoffcurrent_caveat(preset.name), ...
            'source_implementation', ...
                'analysis/Olhoff/+impl/architecture/olhoffSolve.m', ...
            'frozen_by_file', 'analysis/Olhoff/olhoffcurrent_presets.m', ...
            'selected_by_file', 'examples/Performance/conference_bench/confbench_olhoff_preset.m', ...
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
criterion = char(valueOr(stop, 'criterion', 'max_change'));
if ~any(strcmp(criterion, {'max_change', 'relative_l2_change'}))
    error('confbench_method_config:UnknownCriterion', ...
        'stop.%s.criterion must be ''max_change'' or ''relative_l2_change'' (got ''%s'').', ...
        methodKey, criterion);
end
if strcmp(criterion, 'relative_l2_change')
    % The frozen tolerances are max|x - x_old| values; a relative criterion
    % has no production tolerance, so every one it uses must be given.
    if isfield(mcfg.optimization, 'yuksel'); need = {'stage1Tol', 'stage2Tol'};
    else;                                    need = {'tol'}; end
    missing = need(~cellfun(@(n) hasValue(stop, n), need));
    if ~isempty(missing)
        error('confbench_method_config:RelativeTolRequired', ...
            'stop.%s.criterion = ''relative_l2_change'' requires stop.%s.%s.', ...
            methodKey, methodKey, strjoin(missing, sprintf(' and stop.%s.', methodKey)));
    end
    % Set only when it departs from the native rule, so a production
    % configuration stays field-for-field what it was.
    mcfg.optimization.stop_criterion = criterion;
end
if hasValue(stop, 'tol'); mcfg.optimization.convergence_tol = stop.tol; end
if hasValue(stop, 'maxIters')
    mcfg.optimization.max_iters = stop.maxIters;
    if isfield(mcfg.optimization, 'yuksel')
        mcfg.optimization.yuksel.stage1_max_iters = stop.maxIters;   % per stage
    end
end
if hasValue(stop, 'stage1Tol'); mcfg.optimization.yuksel.stage1_tol = stop.stage1Tol; end
if hasValue(stop, 'stage2Tol')
    % stage2_tol is the rule Yuksel applies; convergence_tol mirrors it in
    % every frozen configuration and is kept equal to it.
    mcfg.optimization.yuksel.stage2_tol = stop.stage2Tol;
    mcfg.optimization.convergence_tol = stop.stage2Tol;
end
mcfg.meta.profile_id = profileId;
mcfg.meta.frozen_by = 'examples/Performance/benchmark_profile/profile_freeze_manifest.json';
mcfg.meta.source_implementation = char(profile.source_implementation);
mcfg.meta.threads_per_run = 1;
mcfg.postprocessing.visualize_live = false;
mcfg.postprocessing.save_final_image = false;
mcfg.postprocessing.save_snapshot_image = false;
if nargin >= 4 && ~isempty(outputDir)
    mcfg.meta.output_dir = char(outputDir);
end
end

function tf = hasValue(s, name)
tf = isfield(s, name) && ~isempty(s.(name));
end

function v = valueOr(s, name, default)
if hasValue(s, name); v = s.(name); else; v = default; end
end
