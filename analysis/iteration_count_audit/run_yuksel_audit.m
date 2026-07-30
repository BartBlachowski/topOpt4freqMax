function run_yuksel_audit()
%RUN_YUKSEL_AUDIT Reproduce selected Yuksel runs with opt-in diagnostics.
%
% Baseline cases use the exact Yuksel arguments assembled by
% examples/Performance/performance_comparison.m and run_topopt_from_json.m.
% The final three 320x40 cases are explicitly labeled causal ablations.

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
addpath(fullfile(repoRoot, 'analysis', 'YukselApproach', 'Matlab'));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

outDir = fullfile(thisDir, 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

cases = struct( ...
    'tag', {'baseline_160x20', 'baseline_240x30', 'baseline_320x40', ...
            'baseline_400x50', 'fixed_physical_radius_320x40', ...
            'frozen_mode_320x40', 'frozen_load_320x40'}, ...
    'nelx', {160, 240, 320, 400, 320, 320, 320}, ...
    'nely', {20, 30, 40, 50, 40, 40, 40}, ...
    'rmin', {2, 2, 2, 2, 4, 2, 2}, ...
    'freezeMode', {false, false, false, false, false, true, false}, ...
    'freezeLoad', {false, false, false, false, false, false, true});

summary = table('Size', [numel(cases), 10], ...
    'VariableTypes', {'string','double','double','double','logical','logical', ...
                      'double','double','double','double'}, ...
    'VariableNames', {'Tag','Nelx','Nely','RminElem','FreezeMode','FreezeLoad', ...
                      'Stage1Iterations','Stage2Iterations','TotalIterations','Omega1'});

for k = 1:numel(cases)
    c = cases(k);
    fprintf('\n[AUDIT] %s\n', c.tag);

    runCfg = struct();
    runCfg.E0 = 1.0e7;
    runCfg.Emin = 10.0;          % E0 * E_min_ratio (1e-6)
    runCfg.nu = 0.3;
    runCfg.rho0 = 1.0;
    runCfg.rho_min = 1.0e-6;
    runCfg.beamL = 8.0;
    runCfg.beamH = 1.0;
    runCfg.conv_tol = 3.0e-3;
    runCfg.approach_name = 'Yuksel';
    runCfg.visualize_live = false;
    runCfg.save_frq_iterations = false;
    runCfg.final_modes = 3;
    runCfg.audit_collect = true;
    runCfg.audit_snapshot_every = 10;
    runCfg.audit_freeze_mode = c.freezeMode;
    runCfg.audit_freeze_load = c.freezeLoad;

    [xFinal, uFinal, info] = top99neo_inertial_freq( ...
        c.nelx, c.nely, 0.5, 3.0, c.rmin, 1, 'N', 0.5, 1.0, 0.2, ...
        10000, 200, 'simply', 0, runCfg);

    result = struct();
    result.case = c;
    result.runCfg = runCfg;
    result.xFinal = xFinal;
    result.uFinal = uFinal;
    result.info = info;
    result.generatedAt = char(datetime('now', 'TimeZone', 'Europe/Warsaw', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    save(fullfile(outDir, [c.tag '.mat']), 'result', '-v7.3');

    summary.Tag(k) = string(c.tag);
    summary.Nelx(k) = c.nelx;
    summary.Nely(k) = c.nely;
    summary.RminElem(k) = c.rmin;
    summary.FreezeMode(k) = c.freezeMode;
    summary.FreezeLoad(k) = c.freezeLoad;
    summary.Stage1Iterations(k) = info.timing.stage1_iterations;
    summary.Stage2Iterations(k) = info.timing.stage2_iterations;
    summary.TotalIterations(k) = info.timing.total_iterations;
    summary.Omega1(k) = info.stage2.omega1;
    writetable(summary(1:k,:), fullfile(outDir, 'run_summary.csv'));
end

disp(summary);
end
