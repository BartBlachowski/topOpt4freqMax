function bg_extract_campaigns()
%BG_EXTRACT_CAMPAIGNS  Extract bimodality-gap evidence from the two RECORDED nine-mesh
%   campaigns (read-only) and from the recorded move/stop diagnostic CSVs.
%
%   Sources (never modified):
%     P: examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/runs/<mesh>/SOLVER_RESULT.mat
%        (resTap = the olhoffSolve result: rho, hist, aux.Mnd per iteration, omega, log)
%     S: examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat
%        (records(method_key=='olhoff').x = final design; stopping = summary; no histories)
%   Output: docs/bimodality_gap/data/campaign_metrics.csv, per-iteration CSVs, rho copies.
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
addpath(here);
dataDir = fullfile(fileparts(here), 'data');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
a = 8; b = 1; Rphys = 0.06;

rows = {};
% ---- P: Pedersen + linear mass, adaptive box (production) ---------------
pdir = fullfile(root, 'examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/runs');
meshes = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
for i = 1:size(meshes,1)
    nelx = meshes(i,1); nely = meshes(i,2);
    S = load(fullfile(pdir, sprintf('%dx%d', nelx, nely), 'SOLVER_RESULT.mat'));
    res = S.resTap;
    rho = double(res.rho(:));
    m = bg_metrics(rho, nelx, nely, a, b, Rphys*nely);
    m.formulation = 'P_pedersen_adaptive'; m.nelx = nelx; m.nely = nely;
    m.nOuter = numel(res.hist.N); m.status = res.status;
    m.omega1 = res.omega(1); m.omega2 = res.omega(2); m.gap12 = (res.omega(2)-res.omega(1))/res.omega(1);
    m.eps = 0.05*sqrt(nelx*nely/3200);
    m.final_l2 = res.hist.dxNorm2(end); m.final_maxabs = res.hist.dxOuter(end);
    m.final_move_max = res.hist.move(end); m.final_move_mean = res.aux.moveMean(end);
    m.frac_iters_N2 = mean(res.hist.N >= 2);
    m.last_iter_gap_lt_005 = max([0, find(res.hist.gap12 < 0.05, 1, 'last')]);
    m.Mnd_min_hist = min(res.aux.Mnd); m.Mnd_argmin = find(res.aux.Mnd == min(res.aux.Mnd), 1);
    m.Mnd_slope_last20 = (res.aux.Mnd(end) - res.aux.Mnd(max(1,end-20))) / min(20, m.nOuter-1);
    m.inner_total = sum(res.hist.nInner);
    m.source = sprintf('nine_mesh_pedersen_b21483b/runs/%dx%d/SOLVER_RESULT.mat', nelx, nely);
    rows{end+1} = m; %#ok<AGROW>
    % per-iteration CSV
    T = table((1:m.nOuter)', res.hist.omega(1,:)', res.hist.omega(2,:)', res.hist.N(:), res.hist.gap12(:), ...
        res.hist.dxNorm2(:), res.hist.dxOuter(:), res.hist.move(:), res.aux.moveMean(:), res.hist.nInner(:), ...
        res.hist.vol(:), res.aux.Mnd(:), res.hist.beta(:), ...
        'VariableNames', {'k','omega1','omega2','N','gap12','dxNorm2','dxMax','moveMax','moveMean','nInner','vol','Mnd','beta'});
    writetable(T, fullfile(dataDir, sprintf('iter_P_%dx%d.csv', nelx, nely)));
    save(fullfile(dataDir, sprintf('rho_P_%dx%d.mat', nelx, nely)), 'rho', 'nelx', 'nely', '-v7');
    fprintf('P %dx%d: Mnd=%.4f gray=%.4f L=%.3f w=%.4f (%.2f el) nOuter=%d\n', nelx, nely, m.Mnd, m.gray_01_09, m.L_iso05, m.w_gray_phys, m.w_gray_el, m.nOuter);
end

% ---- S: SIMP + eq.(4b), beta-stall ladder (historical) ------------------
L = load(fullfile(root, 'examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat'), 'records');
rec = L.records(strcmp({L.records.method_key}, 'olhoff'));
for i = 1:numel(rec)
    r = rec(i); nelx = r.mesh(1); nely = r.mesh(2);
    rho = double(r.x(:));
    m = bg_metrics(rho, nelx, nely, a, b, Rphys*nely);
    m.formulation = 'S_simp_ladder'; m.nelx = nelx; m.nely = nely;
    m.nOuter = r.stopping.outer_iterations; m.status = r.status;
    m.omega1 = r.omega(1); m.omega2 = r.omega(2); m.gap12 = (r.omega(2)-r.omega(1))/r.omega(1);
    m.eps = r.stopping.eps_l2; m.final_l2 = r.stopping.final_l2_density_change; m.final_maxabs = r.stopping.final_max_density_change;
    m.final_move_max = r.stopping.final_move_limit; m.final_move_mean = NaN;
    m.frac_iters_N2 = NaN; m.last_iter_gap_lt_005 = NaN; m.Mnd_min_hist = NaN; m.Mnd_argmin = NaN; m.Mnd_slope_last20 = NaN;
    m.inner_total = NaN;
    m.source = sprintf('campaign_9mesh_r2/benchmark_records.mat (olhoff %dx%d, cfg %s)', nelx, nely, r.effective_config_hash(1:12));
    rows{end+1} = m; %#ok<AGROW>
    save(fullfile(dataDir, sprintf('rho_S_%dx%d.mat', nelx, nely)), 'rho', 'nelx', 'nely', '-v7');
    fprintf('S %dx%d: Mnd=%.4f gray=%.4f L=%.3f w=%.4f (%.2f el) nOuter=%d stage=%d\n', nelx, nely, m.Mnd, m.gray_01_09, m.L_iso05, m.w_gray_phys, m.w_gray_el, m.nOuter, r.stopping.final_ladder_stage);
end
bg_write_rows(rows, fullfile(dataDir, 'campaign_metrics.csv'));
end
