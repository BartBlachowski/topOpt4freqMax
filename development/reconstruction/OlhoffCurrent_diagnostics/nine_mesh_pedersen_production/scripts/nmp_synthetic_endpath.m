function S = nmp_synthetic_endpath(smokeRecordsMat, outJson)
%NMP_SYNTHETIC_ENDPATH  Software-mechanics test of the END of a full Olhoff-only
%   nine-mesh run of performance_comparison.m, with NO solve.
%
%   The runner saves benchmark_records.mat only AFTER confbench_scaling_fit and
%   confbench_export succeed, and those two are not inside try/catch.  An
%   Olhoff-only, nine-row, performance_campaign = 1 record set has never passed
%   through them.  This builds nine synthetic records from the smoke record (the
%   design vector is nearest-neighbour resampled to each mesh, status set to
%   NATIVE_CONVERGED, ok = true) and runs, in a temporary directory, the exact
%   end-of-script sequence: scaling fit, manifest, export, -v7.3 save, complexity
%   plots, topology images.  Nothing here is evidence about the science.
here = fileparts(mfilename('fullpath'));
D = fileparts(here);
oc = fileparts(fileparts(D));
repo = fileparts(fileparts(oc));
perf = fullfile(repo, 'examples', 'Performance');
restoredefaultpath;
addpath(perf); addpath(fullfile(perf, 'conference_bench'));
addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'analysis', 'three_method_parametric_study'));
addpath(oc); addpath(here);
olhoffcurrent_scrub_forbidden_paths(repo);

L = load(smokeRecordsMat);
rec0 = L.records(1);
M = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
od = fullfile(tempname(), 'nmp_synthetic_endpath');
mkdir(od);

cfg = L.cfg;
cfg.resolutions = M;
cfg.maxOuterOverride = [];
cfg.scientificEvidence = true;
cfg.performanceCampaign = true;
cfg.outputDir = od;
cfg.runLabel = 'nmp_synthetic_endpath';
cfg.methods = struct('proposed', false, 'yuksel', false, 'olhoff', true);

x0 = reshape(rec0.x, rec0.mesh(2), rec0.mesh(1));
records = struct([]);
methodConfigs = cell(size(M, 1), 1);
for r = 1:size(M, 1)
    nelx = M(r, 1); nely = M(r, 2);
    methodConfigs{r, 1} = confbench_method_config('olhoff', nelx, nely, od);
    rec = rec0;
    iy = min(rec0.mesh(2), max(1, ceil((1:nely) * rec0.mesh(2) / nely)));
    ix = min(rec0.mesh(1), max(1, ceil((1:nelx) * rec0.mesh(1) / nelx)));
    rec.x = reshape(x0(iy, ix), [], 1);
    rec.mesh = [nelx nely];
    rec.n_elements = nelx*nely;
    rec.status = 'NATIVE_CONVERGED';
    rec.ok = true;
    rec.overridden = false;
    rec.is_warmup = false;
    rec.scientific_observation = true;
    scale = nelx*nely/3200;
    rec.times.total_wall_time_s = rec0.times.total_wall_time_s*scale;
    rec.times.total_wall_time_per_outer_s = rec0.times.total_wall_time_per_outer_s*scale;
    rec.times.outer_time_excluding_inner_per_outer_mean_s = rec0.times.outer_time_excluding_inner_per_outer_mean_s*scale;
    rec.times.eigen_time_per_outer_mean_s = rec0.times.eigen_time_per_outer_mean_s*scale;
    rec.times.inner_time_per_outer_mean_s = rec0.times.inner_time_per_outer_mean_s*scale;
    rec.times.inner_time_per_inner_iteration_mean_s = rec0.times.inner_time_per_inner_iteration_mean_s*scale;
    rec = orderfields(rec);
    if isempty(records); records = rec; else; records(end+1) = rec; end %#ok<AGROW>
end

S = struct('schema', 'nmp_synthetic_endpath/1', 'when', nmp_now(), 'output_dir', od, 'steps', struct([]));
step = @(name, ok, msg) struct('name', name, 'ok', ok, 'message', msg);
try
    scaling = confbench_scaling_fit(cfg, records);
    S.steps = [S.steps, step('confbench_scaling_fit', scaling.fitted, sprintf('fitted=%d', scaling.fitted))];
    resolvedImpl = struct('olhoff', records(1).resolved_implementation, ...
        'run_topopt_from_json', which('run_topopt_from_json'), ...
        'study_evaluate_design', which('study_evaluate_design'), ...
        'study_base_config', which('study_base_config'));
    manifest = confbench_manifest(cfg, methodConfigs, resolvedImpl);
    manifest.warmup = L.manifest.warmup;
    manifest.preflight = L.manifest.preflight;
    capRows = arrayfun(@(r) strcmp(r.status, 'CAP_HIT'), records);
    manifest.cap_summary = struct('any_cap_hit', any(capRows), 'n_cap_hit', sum(capRows), ...
        'n_records', numel(records), 'cap_hit_rows', {{}}, 'meaning', 'synthetic');
    manifest.path_scrub = struct();
    manifest.path_scrub.removed_entries = {};
    S.steps = [S.steps, step('confbench_manifest', true, '')];
    files = confbench_export(cfg, records, manifest, scaling);
    S.steps = [S.steps, step('confbench_export', true, strjoin(struct2cell(files)', '; '))];
    save(fullfile(od, 'benchmark_records.mat'), 'records', 'cfg', 'manifest', 'scaling', '-v7.3');
    S.steps = [S.steps, step('save_v73', exist(fullfile(od, 'benchmark_records.mat'), 'file') == 2, '')];
    plotFiles = confbench_complexity_plots(cfg, records, scaling);
    S.steps = [S.steps, step('confbench_complexity_plots', true, strjoin(struct2cell(plotFiles)', '; '))];
    topo = confbench_topology_images(cfg, records);
    S.steps = [S.steps, step('confbench_topology_images', topo.n_written == 9, sprintf('written=%d skipped=%d', topo.n_written, topo.n_skipped))];
catch ME
    S.steps = [S.steps, step('EXCEPTION', false, getReport(ME, 'extended', 'hyperlinks', 'off'))];
end
S.pass = ~isempty(S.steps) && all([S.steps.ok]) && numel(S.steps) == 6;
fid = fopen(outJson, 'w'); fprintf(fid, '%s\n', jsonencode(S, 'PrettyPrint', true)); fclose(fid);
fprintf('SYNTHETIC END PATH PASS=%d\n', S.pass);
for i = 1:numel(S.steps); fprintf('  [%d] %s\n', S.steps(i).ok, S.steps(i).name); end
end
