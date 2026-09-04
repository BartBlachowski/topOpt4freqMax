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
%   See also PERFORMANCE_COMPARISON, CONFBENCH_MANIFEST, OLHOFFM4_VERIFY_IMPORT.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));

pre = struct('checks', struct('name', {}, 'pass', {}, 'detail', {}), 'pass', true);

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
ok = cfg.scientificEvidence == (isempty(cfg.maxOuterOverride) && all(NE >= 3200));
pre = add(pre, 'scientific_evidence derived from cfg, not declared', ok, ...
    sprintf('scientific_evidence=%d (min NE = %d, floor 3200 = 160x20; override=%s)', ...
        cfg.scientificEvidence, min(NE), mat2str(cfg.maxOuterOverride)));

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

% ---- 5. the imported Du-Olhoff (M4) reconstruction ----------------------
if cfg.methods.olhoff
    imp = olhoffm4_verify_import('Verbose', false);
    pre = add(pre, 'imported Olhoff files match IMPORT_MANIFEST.json', ...
        isempty(imp.imported_hash_mismatches), strjoin(imp.imported_hash_mismatches, '; '));
    pre = add(pre, 'the audited source repository is unchanged since import', ...
        isempty(imp.source_hash_mismatches), strjoin(imp.source_hash_mismatches, '; '));
    pre = add(pre, 'no undeclared modification to the imported solver', ...
        isempty(imp.undeclared_modifications), ...
        sprintf('declared: %s', strjoin(imp.declared_modifications, ', ')));
    pre = add(pre, 'Olhoff dispatch gate resolves inside the import', ...
        imp.dispatch_ok, resolvedSummary(imp));
    pre.olhoff_import = imp;

    % ---- 5b. the frozen realization, field by field ---------------------
    for r = 1:size(R,1)
        c = olhoffm4_config(R(r,1), R(r,2));
        tag = sprintf('%dx%d', R(r,1), R(r,2));
        pre = add(pre, ['Olhoff ' tag ': genuine nested MMA sub-optimization'], ...
            strcmp(c.innerSolver,'mma') && strcmp(c.innerVar,'drho') && ...
            strcmp(c.mmaVariant,'published') && c.offDiag, ...
            sprintf('innerSolver=%s innerVar=%s mmaVariant=%s offDiag=%d', ...
                c.innerSolver, c.innerVar, c.mmaVariant, c.offDiag));
        pre = add(pre, ['Olhoff ' tag ': M4 multiplicity treatment, frozen subN'], ...
            strcmp(c.multRule,'subspace') && c.subN == 2, ...
            sprintf('multRule=%s subN=%d (no threshold classifier)', c.multRule, c.subN));
        rminEl = c.rminPhys/(c.b/c.nely);
        pre = add(pre, ['Olhoff ' tag ': fixed physical filter R = 0.06'], ...
            c.rminPhys == 0.06 && isnan(c.rminEl) && abs(rminEl - 0.06*c.nely) < 1e-12, ...
            sprintf('rminPhys=%.10g -> rminEl=%.10g (derived at run time), filterMode=%s', ...
                c.rminPhys, rminEl, c.filterMode));
        pre = add(pre, ['Olhoff ' tag ': tolInner = 0.05'], c.tolInner == 0.05, ...
            sprintf('tolInner=%.10g maxInner=%d minInner=%d', c.tolInner, c.maxInner, c.minInner));
        epsRms = c.tolOuter/sqrt(c.nelx*c.nely);
        pre = add(pre, ['Olhoff ' tag ': outer RMS stopping semantics'], ...
            strcmp(c.outerNorm,'l2') && abs(epsRms - 0.05/sqrt(3200)) < 1e-15 && ...
            strcmp(c.outerGuard,'settledmove'), ...
            sprintf(['||drho||_2 < %.10g, i.e. per-element RMS < %.9e (constant ' ...
                'across meshes); guard=%s'], c.tolOuter, epsRms, c.outerGuard));
        pre = add(pre, ['Olhoff ' tag ': S2 continuation realization as frozen'], ...
            strcmp(c.moveFamily,'S2') && isequal(c.s2Levels,[0.04 0.02 0.01 0.005]) && ...
            c.move == 0.04 && c.s2Window == 10 && c.s2Tol == 5e-3 && ~isfield(c,'s2Signal'), ...
            sprintf(['moveFamily=S2 move0=%.4g ladder=%s window=%d tol=%.4g; ' ...
                's2Signal absent => legacy beta signal (the design-driven ' ...
                '''drms'' trigger was measured and NOT adopted)'], ...
                c.move, mat2str(c.s2Levels), c.s2Window, c.s2Tol));
        pre = add(pre, ['Olhoff ' tag ': single thread and diagnostics off'], ...
            c.threads == 1 && ~c.diag, ...
            sprintf(['threads=%d diag=%d (the per-iteration recorder is proved ' ...
                'bitwise inert and is off so it is not timed)'], c.threads, c.diag));
    end
end

% ---- 6. no superseded Olhoff implementation is reachable ----------------
forbidden = olhoffm4_forbidden_paths();
onPath = strsplit(path, pathsep);
hits = {};
for i = 1:numel(forbidden)
    p = fullfile(repo, forbidden{i});
    for k = 1:numel(onPath)
        if strncmp(onPath{k}, p, numel(p)); hits{end+1} = onPath{k}; end %#ok<AGROW>
    end
end
pre = add(pre, 'no superseded Olhoff directory is on the MATLAB path', isempty(hits), ...
    strjoin(hits, '; '));

names = {'olhoffOpt','model2D','assemble2D','eigSolve','genGrad','innerLoop', ...
         'prepFilter','applyFilter','multRule','moveControl','deltaLambda'};
leaks = {};
for i = 1:numel(names)
    f = which(names{i});
    for k = 1:numel(forbidden)
        p = fullfile(repo, forbidden{k});
        if ~isempty(f) && strncmp(f, p, numel(p))
            leaks{end+1} = sprintf('%s -> %s', names{i}, f); %#ok<AGROW>
        end
    end
end
pre = add(pre, 'no Olhoff-family name resolves into a superseded tree', isempty(leaks), ...
    strjoin(leaks, '; '));

% ---- 7. the dispatched methods ------------------------------------------
pre = add(pre, 'run_topopt_from_json is the repository tool copy', ...
    strcmp(which('run_topopt_from_json'), fullfile(repo,'tools','Matlab','run_topopt_from_json.m')), ...
    which('run_topopt_from_json'));
pre = add(pre, 'the common E1/E2/E3 evaluator is the frozen study copy', ...
    strcmp(which('study_evaluate_design'), ...
        fullfile(repo,'analysis','three_method_parametric_study','study_evaluate_design.m')), ...
    which('study_evaluate_design'));

% ---- 8. frozen profile identity for Proposed and Yuksel ----------------
fp = fullfile(repo,'analysis','three_method_parametric_study','results','profile_freeze_manifest.json');
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

% ---- 11. the reporting contract ----------------------------------------
sch = confbench_timing_schema();
pre = add(pre, 'timing schema is defined for every enabled method', ...
    isfield(sch.methods,'Proposed') && isfield(sch.methods,'Yuksel') && ...
    isfield(sch.methods,'DuOlhoffReconstructionM4'), sch.cross_method_warning);
cav = confbench_caveats();
pre = add(pre, 'the Du-Olhoff caveat is defined and non-empty', ...
    ischar(cav.olhoff) && numel(cav.olhoff) > 100, cav.olhoff);

pre.pass = all([pre.checks.pass]);
end

% =========================================================================
function pre = add(pre, name, ok, detail)
pre.checks(end+1) = struct('name', name, 'pass', logical(ok), 'detail', char(string(detail)));
end

function s = meshList(R)
parts = arrayfun(@(i) sprintf('%dx%d', R(i,1), R(i,2)), 1:size(R,1), 'UniformOutput', false);
s = strjoin(parts, ', ');
end

function s = resolvedSummary(imp)
if ~imp.dispatch_ok
    s = imp.dispatch_error;
else
    s = sprintf('%d owned functions resolve inside analysis/OlhoffM4Reconstruction', ...
        numel(imp.resolved));
end
end
