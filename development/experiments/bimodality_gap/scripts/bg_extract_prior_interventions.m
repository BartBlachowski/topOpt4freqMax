function bg_extract_prior_interventions()
%BG_EXTRACT_PRIOR_INTERVENTIONS  Extract the RECORDED fixed-move interventions on the
%   SIMP + eq.(4b) formulation (read-only sources):
%     analysis/OlhoffCurrent/evidence/move_activity_400/{F400,P400}_400x50_trajectory.mat
%        F400: fixed move 0.04, stopped by the native rule at 369 (RHO 20000 x 369)
%        P400: production ladder, converged at 139               (RHO 20000 x 139)
%     analysis/OlhoffCurrent/diagnostics/move_stop/runs/*_iterations.csv
%        baseline (ladder) and fixed-move 0.04 at 160x20 and 320x40, per-iteration Mnd
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
addpath(here);
dataDir = fullfile(fileparts(here), 'data');
a = 8; b = 1; Rphys = 0.06;
rows = {};
src = fullfile(root, 'analysis/OlhoffCurrent/evidence/move_activity_400');
for f = {'F400', 'P400'}
    S = load(fullfile(src, [f{1} '_400x50_trajectory.mat']));
    nelx = S.meta.nelx; nely = S.meta.nely; NE = nelx*nely;
    rho = S.RHO(:, end);
    m = bg_metrics(rho, nelx, nely, a, b, Rphys*nely);
    m.formulation = ['S_simp_' lower(f{1})]; m.arm = S.meta.arm; m.label = S.meta.label;
    m.nelx = nelx; m.nely = nely; m.nOuter = size(S.RHO, 2);
    m.omega1 = S.hist.omega(1, end); m.omega2 = S.hist.omega(2, end);
    m.gap12 = (m.omega2 - m.omega1)/m.omega1;
    m.final_move_max = S.move(end);
    m.preset = S.meta.preset; m.overrides = jsonencode(S.meta.overrides);
    m.source = sprintf('evidence/move_activity_400/%s_400x50_trajectory.mat', f{1});
    rows{end+1} = m; %#ok<AGROW>
    Mnd = 4*mean(S.RHO.*(1-S.RHO), 1)';
    gray = mean(S.RHO > 0.1 & S.RHO < 0.9, 1)';
    T = table((1:m.nOuter)', S.hist.omega(1,:)', S.hist.omega(2,:)', S.hist.N(:), S.hist.gap12(:), ...
        S.hist.dxNorm2(:), S.hist.dxOuter(:), S.move(:), S.hist.nInner(:), S.hist.vol(:), Mnd, gray, S.hist.stage(:), ...
        'VariableNames', {'k','omega1','omega2','N','gap12','dxNorm2','dxMax','move','nInner','vol','Mnd','gray','stage'});
    writetable(T, fullfile(dataDir, sprintf('iter_S_%s_400x50.csv', lower(f{1}))));
    save(fullfile(dataDir, sprintf('rho_S_%s_400x50.mat', lower(f{1}))), 'rho', 'nelx', 'nely', '-v7');
    fprintf('%s 400x50 (%s): nOuter=%d Mnd=%.4f gray=%.4f w=%.4f (%.2f el) omega1=%.4f\n', f{1}, S.meta.label, m.nOuter, m.Mnd, m.gray_01_09, m.w_gray_phys, m.w_gray_el, m.omega1);
end
bg_write_rows(rows, fullfile(dataDir, 'prior_intervention_metrics.csv'));
% copy the move_stop per-iteration CSVs (small) for self-containment
ms = fullfile(root, 'analysis/OlhoffCurrent/diagnostics/move_stop/runs');
for f = {'baseline_160x20','baseline_320x40','fixedmove_160x20','fixedmove_320x40'}
    copyfile(fullfile(ms, [f{1} '_iterations.csv']), fullfile(dataDir, ['iter_S_movestop_' f{1} '.csv']));
end
end
