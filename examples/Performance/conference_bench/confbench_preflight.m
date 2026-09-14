function pre = confbench_preflight(cfg, methodConfigs)
%CONFBENCH_PREFLIGHT  Refuse to start unless everything about the run is proved.
%
%   pre = CONFBENCH_PREFLIGHT(cfg, methodConfigs) runs every check that can be
%   made WITHOUT solving anything, and returns a report.  pre.pass is true only
%   when every check passed.  The driver stops on a false.
%
%   The checks exist because each one has been wrong at least once in this
%   project's history: a superseded Olhoff implementation dispatched silently, a
%   configuration that drifted from its frozen source, a mesh list that came
%   from a manifest instead of the script, a memory sampler inside a timed loop.
%
%   See also PERFORMANCE_COMPARISON, CONFBENCH_MANIFEST,
%            OLHOFFCURRENT_ASSERT_DISPATCH, OLHOFFCURRENT_CURRENTNESS.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));

pre = struct('checks', struct('name', {}, 'pass', {}, 'detail', {}), ...
             'notes', {{}}, 'pass', true);

% ---- 1. configuration shape --------------------------------------------
R = cfg.resolutions;
ok = isnumeric(R) && ismatrix(R) && size(R,2) == 2 && ~isempty(R) && ...
     all(R(:) > 0) && all(mod(R(:),1) == 0);
pre = add(pre, 'cfg.resolutions is a non-empty N-by-2 integer matrix', ok, ...
    sprintf('%d mesh(es): %s', size(R,1), meshList(R)));

ok = all(mod(R(:,2), 2) == 0);
pre = add(pre, 'every nely is even (mid-height supports)', ok, ...
    sprintf('odd nely at row(s): %s', mat2str(find(mod(R(:,2),2) ~= 0).')));

ok = size(unique(R, 'rows'), 1) == size(R, 1);
pre = add(pre, 'no duplicate resolutions', ok, '');

ok = any(cell2mat(struct2cell(cfg.methods)));
pre = add(pre, 'at least one method enabled', ok, ...
    sprintf('proposed=%d yuksel=%d olhoff=%d', cfg.methods.proposed, ...
            cfg.methods.yuksel, cfg.methods.olhoff));

% ---- 2. run class is DERIVED from the configuration, not asserted -------
NE = R(:,1).*R(:,2);
% A RAISED Yuksel safety budget leaves the run scientific; a LOWERED one
% truncates the method before its own stopping rule can fire, exactly like
% maxOuterOverride, and must therefore void the run the same way.
yukselFrozen = confbench_frozen_budget('yuksel');
yukselUsed   = yukselFrozen;
if isfield(cfg, 'yukselMaxIters') && ~isempty(cfg.yukselMaxIters)
    yukselUsed = cfg.yukselMaxIters;
end
ok = cfg.scientificEvidence == (isempty(cfg.maxOuterOverride) && all(NE >= 3200) ...
    && yukselUsed >= yukselFrozen);
pre = add(pre, 'scientific_evidence derived from cfg, not declared', ok, ...
    sprintf(['scientific_evidence=%d (min NE = %d, floor 3200 = 160x20; override=%s; ' ...
             'yuksel budget %d vs frozen %d)'], ...
        cfg.scientificEvidence, min(NE), mat2str(cfg.maxOuterOverride), ...
        yukselUsed, yukselFrozen));

ok = ~cfg.performanceCampaign || cfg.scientificEvidence;
pre = add(pre, 'a performance campaign is also scientific evidence', ok, ...
    sprintf('performance_campaign=%d', cfg.performanceCampaign));

% ---- 3. expensive-run acknowledgement ----------------------------------
big = NE > 3200;
ok = ~any(big) || cfg.confirmLongCampaign;
pre = add(pre, 'meshes above 160x20 are explicitly acknowledged', ok, ...
    sprintf(['%d mesh(es) exceed 160x20. Set cfg.confirmLongCampaign = true ' ...
        'in performance_comparison.m to run them.'], sum(big)));

% ---- 4. single-thread operation ----------------------------------------
ok = ~cfg.singleThread || maxNumCompThreads() == 1;
pre = add(pre, 'single-threaded execution is in force', ok, ...
    sprintf('maxNumCompThreads = %d, requested single thread = %d', ...
        maxNumCompThreads(), cfg.singleThread));

% ---- 5. the production Du-Olhoff implementation, analysis/OlhoffCurrent -
if cfg.methods.olhoff
    % The identity of the running solver is proved from evidence held inside
    % THIS repository -- an integrity manifest over the promoted source, the
    % recorded upstream provenance, and a path gate that refuses to return
    % unless exactly one Olhoff implementation is visible.  The external
    % development repository is outside this repository and not under its
    % control; its state is recorded as a note below, NEVER as a gate, and it
    % is forbidden from the production path entirely.
    ocGuard = olhoffcurrent_paths(); %#ok<NASGU>

    man = olhoffcurrent_source_manifest();
    pre = add(pre, 'promoted Olhoff source hashes to SOURCE_MANIFEST.json', man.ok, ...
        joinOr([man.mismatches, man.missing, man.extra], ...
            sprintf('%d files re-hashed; tree %s', man.nFiles, man.treeHash)));

    cur = olhoffcurrent_currentness('Verbose', false);
    % LOCAL_MODIFIED is a BLOCKER: production source edited in place means the
    % recorded provenance no longer describes the code about to run.
    % UPSTREAM_AHEAD is NOT a blocker -- upstream is a development tree, and
    % production currentness changes only when a state is explicitly promoted.
    pre = add(pre, 'promoted Olhoff source is not modified in place', ...
        ~strcmp(cur.state, 'LOCAL_MODIFIED'), ...
        sprintf('currentness state = %s', cur.state));
    pre = add(pre, 'Olhoff provenance is consistent with its upstream', ...
        ~strcmp(cur.state, 'PROVENANCE_MISMATCH'), cur.detail);
    pre.olhoff_currentness = cur;

    gate = olhoffcurrent_assert_dispatch('Throw', false);
    pre = add(pre, 'exactly one Olhoff implementation is visible to MATLAB', ...
        gate.ok, joinOr(gate.blockers, sprintf(['all %d production-owned symbols ' ...
            'resolve inside analysis/Olhoff/+impl, with no shadow ' ...
            'candidate anywhere else (checked with which -all)'], ...
            numel(gate.resolved))));
    pre.olhoff_dispatch = gate;

    prov = olhoffcurrent_provenance();
    pre = note(pre, sprintf(['production Olhoff: %s, production preset %s (event %s), promoted ' ...
        'from %s %s @ %s'], prov.implementation, prov.production_preset, ...
        prov.production_preset_event_date, prov.source.repository, prov.source.branch, ...
        prov.source.commit));
    pre = note(pre, sprintf('  promoted source tree sha256: %s (%d files)', ...
        prov.live_source_tree_sha256, prov.live_source_n_files));
    if cur.upstream.reachable
        pre = note(pre, sprintf(['  external development repository (PROVENANCE ONLY, ' ...
            'not a gate, FORBIDDEN on the production path): recorded branch %s, head %s, ' ...
            '%d commit(s) ahead of the promoted state'], cur.upstream.branch, ...
            cur.upstream.head, cur.upstream.commitsAhead));
    else
        pre = note(pre, ['  external development repository not present on this ' ...
            'machine; production does not depend on it']);
    end

    % ---- 5a. WHICH preset the benchmark runs --------------------------------
    % The benchmark names its Olhoff preset (confbench_olhoff_preset).  It must be
    % the preset recorded as production in PROVENANCE.json, and production-eligible;
    % the historical stage-exhaustion diagnostic can never pass here.
    benchPreset = confbench_olhoff_preset();
    bp = olhoffcurrent_preset(benchPreset);
    pre = add(pre, 'the benchmark names a canonical, production-eligible Olhoff preset', ...
        strcmp(bp.resolvedVia, 'canonical') && bp.productionEligible, ...
        sprintf('%s (role %s, eligible %d, resolved via %s)', bp.name, bp.role, ...
            bp.productionEligible, bp.resolvedVia));
    pre = add(pre, 'the benchmark Olhoff preset is the recorded production preset', ...
        strcmp(bp.name, prov.production_preset), ...
        sprintf('benchmark %s, PROVENANCE.json production %s', bp.name, prov.production_preset));
    pre.olhoff_preset = bp.name;

    % ---- 5b. the realization, field by field, PER PRESET ---------------------
    % Read from the resolved configuration (canonical) and its flat VIEW, so the
    % assertions check what will actually be solved rather than a separately
    % maintained copy of it.  confbench_olhoff_assertions holds one explicit
    % assertion set per preset; none accepts either formulation.
    for r = 1:size(R,1)
        cc = confbench_method_config('olhoff', R(r,1), R(r,2));
        tag = sprintf('%dx%d', R(r,1), R(r,2));
        pre = add(pre, ['Olhoff ' tag ': method config carries the benchmark preset'], ...
            strcmp(cc.olhoff_preset, bp.name), sprintf('mcfg.olhoff_preset=%s', cc.olhoff_preset));
        chk = confbench_olhoff_assertions(bp.name, cc, tag);
        for q = 1:numel(chk)
            pre = add(pre, chk(q).name, chk(q).pass, chk(q).detail);
        end
    end
end

% ---- 6. no non-production Olhoff implementation is reachable ------------
% The forbidden list covers every historical, experimental and audit tree in
% this repository AND the external development repository by absolute path.
[repoRel, absForbidden] = olhoffcurrent_forbidden_paths();
forbidden = absForbidden(:).';
for i = 1:numel(repoRel); forbidden{end+1} = fullfile(repo, repoRel{i}); end %#ok<AGROW>

onPath = strsplit(path, pathsep);
hits = {};
for i = 1:numel(forbidden)
    for k = 1:numel(onPath)
        if strncmp(onPath{k}, forbidden{i}, numel(forbidden{i}))
            hits{end+1} = onPath{k}; %#ok<AGROW>
        end
    end
end
pre = add(pre, 'no non-production Olhoff directory is on the MATLAB path', isempty(hits), ...
    strjoin(hits, '; '));

% Symbol resolution is checked by olhoffcurrent_assert_dispatch in section 5,
% over EVERY symbol the production tree owns and with which(name,'-all') -- so
% a shadowed second copy is caught even when the winning resolution is correct.
% What follows is a deliberately independent second opinion on the handful of
% names that have actually been mis-dispatched in this project's history.  It
% is written against which(-all) too: a check that looked only at the winner
% would report "clean" for exactly the contamination that matters.
names = {'olhoffOpt','olhoffSolve','model2D','assemble2D','eigSolve','genGrad', ...
         'innerLoop','prepFilter','applyFilter','multRule','moveControl', ...
         'deltaLambda','massScale','mmasub','subsolv','useMMA'};
leaks = {};
for i = 1:numel(names)
    cand = which(names{i}, '-all');
    if ischar(cand); cand = {cand}; end
    for c = 1:numel(cand)
        for k = 1:numel(forbidden)
            if strncmp(cand{c}, forbidden{k}, numel(forbidden{k}))
                leaks{end+1} = sprintf('%s -> %s', names{i}, cand{c}); %#ok<AGROW>
            end
        end
    end
end
pre = add(pre, 'no Olhoff-family name resolves OR SHADOWS into a non-production tree', ...
    isempty(leaks), strjoin(leaks, '; '));

% ---- 7. the dispatched methods ------------------------------------------
pre = add(pre, 'run_topopt_from_json is the repository tool copy', ...
    strcmp(which('run_topopt_from_json'), fullfile(repo,'tools','Matlab','run_topopt_from_json.m')), ...
    which('run_topopt_from_json'));
pre = add(pre, 'the common E1/E2/E3 evaluator is the frozen study copy', ...
    strcmp(which('study_evaluate_design'), ...
        fullfile(repo,'examples','Performance','benchmark_profile','study_evaluate_design.m')), ...
    which('study_evaluate_design'));

% ---- 8. frozen profile identity for Proposed and Yuksel ----------------
fp = fullfile(repo,'examples','Performance','benchmark_profile','profile_freeze_manifest.json');
frozen = jsondecode(fileread(fp));
if cfg.methods.proposed
    [~, pid] = confbench_method_config('proposed', R(1,1), R(1,2));
    pre = add(pre, 'Proposed runs its frozen profile', ...
        strcmp(pid, frozen.profiles.proposed_practical.profile_id), pid);
end
if cfg.methods.yuksel
    [~, pid] = confbench_method_config('yuksel', R(1,1), R(1,2));
    pre = add(pre, 'Yuksel runs its frozen profile', ...
        strcmp(pid, frozen.profiles.yuksel_practical.profile_id), pid);
end

% ---- 9. memory is out of the contract ----------------------------------
% No configuration may re-enable the RSS sampler that run_topopt_from_json
% would otherwise run at 10 Hz INSIDE the timed optimization loop.
% confbench_run_case sets benchmark.measure_memory = false immediately before
% each dispatched call; this check catches a config that turns it back on.
memOn = {};
fn = fieldnames(methodConfigs);
for i = 1:numel(fn)
    mc = methodConfigs.(fn{i});
    if isfield(mc, 'benchmark') && isfield(mc.benchmark, 'measure_memory') && ...
            logical(mc.benchmark.measure_memory)
        memOn{end+1} = fn{i}; %#ok<AGROW>
    end
end
pre = add(pre, 'no memory instrumentation is requested', isempty(memOn), ...
    ['memory is not measured and not reported. ' confbench_caveats().memory]);

% ---- 10. output isolation ----------------------------------------------
legacy = { fullfile(repo,'examples','Performance','benchmark_results.json'), ...
           fullfile(repo,'examples','Performance','table1_performance.csv'), ...
           fullfile(repo,'examples','Performance','final_campaign') };
clash = {};
for i = 1:numel(legacy)
    if strncmp(cfg.outputDir, legacy{i}, numel(legacy{i})); clash{end+1} = legacy{i}; end %#ok<AGROW>
end
pre = add(pre, 'output directory does not overwrite earlier evidence', isempty(clash), ...
    sprintf('outputDir = %s', cfg.outputDir));

% A rerun under the same label lands on top of the previous run's artifacts.
% That is usually what is wanted and is recoverable from git, but it is never
% allowed to happen silently.
priorManifest = fullfile(cfg.outputDir, 'benchmark_manifest.json');
if exist(priorManifest, 'file') == 2
    pre = note(pre, sprintf(['%s already holds a completed run; its artifacts ' ...
        'will be OVERWRITTEN by this one. The previous version is in git, or ' ...
        'set cfg.runLabel to keep both.'], cfg.outputDir));
end

% ---- 11. the reporting contract ----------------------------------------
sch = confbench_timing_schema();
pre = add(pre, 'timing schema is defined for every enabled method', ...
    isfield(sch.methods,'Proposed') && isfield(sch.methods,'Yuksel') && ...
    isfield(sch.methods,'DuOlhoffReconstruction') && ...
    strcmp(sch.methods.DuOlhoffReconstruction.preset, confbench_olhoff_preset()), sch.cross_method_warning);
cav = confbench_caveats();
pre = add(pre, 'the Du-Olhoff caveat is defined and non-empty', ...
    ischar(cav.olhoff) && numel(cav.olhoff) > 100, cav.olhoff);

pre.pass = all([pre.checks.pass]);
end

% =========================================================================
function pre = add(pre, name, ok, detail)
pre.checks(end+1) = struct('name', name, 'pass', logical(ok), 'detail', char(string(detail)));
end

function pre = note(pre, text)
% Recorded and printed, but never part of pre.pass.  Notes carry provenance
% that a reader needs to see and that must not be able to block a run.
pre.notes{end+1} = char(string(text));
end

function s = joinOr(findings, okText)
% The findings when there are any, otherwise what was actually proved.
if isempty(findings); s = okText; else; s = strjoin(findings, '; '); end
end

function s = meshList(R)
parts = arrayfun(@(i) sprintf('%dx%d', R(i,1), R(i,2)), 1:size(R,1), 'UniformOutput', false);
s = strjoin(parts, ', ');
end

