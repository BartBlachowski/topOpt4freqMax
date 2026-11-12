function T = tables5_8_fig10_11_alpha_sweep(quick)
%TABLES5_8_FIG10_11_ALPHA_SWEEP  Sections 4.2-4.3: two-mode load weighting, Tables 5-8 and
%   Figures 10-11.
%
%   T = tables5_8_fig10_11_alpha_sweep()       clamped beam 400x50 and building 80x240,
%                                              alpha = 1, 0.75, 0.5, 0.25, 0 (~5 min)
%   T = tables5_8_fig10_11_alpha_sweep(true)   quick: 160x20 and 40x120
%
%   Objective (Eq. (14)): alpha c_1 + (1 - alpha) c_2 with the loads of the first two
%   solid-domain modes.  Reported per design: the lowest two structural frequencies
%   (Tables 5 and 7) and the tracked frequencies (Tables 6 and 8): for each reference
%   shape phi_j^(0), the structural mode among the 100 lowest eigenpairs with the largest
%   mass-weighted MAC (Eq. (9)), its position, MAC and gain relative to the starting
%   design.  Output: results/alpha_sweep/*.csv and the topology images.
if nargin < 1, quick = false; end
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('alpha_sweep');
alphas = [1 0.75 0.5 0.25 0];
if quick, probs = {problem_clamped_beam(160, 20), problem_building(40, 120)};
else,     probs = {problem_clamped_beam(), problem_building()}; end
paperOmega1 = {[368.24 288.83 259.52 231.06 115.87], [87.11 56.00 45.37 35.27 35.95]};  % Tables 5 and 7
rows = {};
for ip = 1:numel(probs)
    prob = probs{ip};
    mdl = fe_mesh(prob);
    % starting design (reference of the gains): passive frame solid, rest uniform at V_f
    x0 = prob.volfrac*ones(mdl.nEl, 1);
    if ~isempty(mdl.pas), x0(mdl.pas) = 1; x0(mdl.act) = (prob.volfrac*mdl.nEl - numel(mdl.pas))/numel(mdl.act); end
    [K, M] = assemble_KM(mdl, x0, prob.E, 1e-6*prob.E, prob.rho, 1e-6);  f = mdl.free;
    lam0 = sort(real(eigs(K(f,f), M(f,f), 2, 'smallestabs')));  omIni = sqrt(lam0);
    for ia = 1:numel(alphas)
        P = proposed_freq(prob, struct('alpha', alphas(ia)));
        [om, ~, info] = structural_modes(prob, P.x, 2, struct('mdl', mdl));
        tr = trackModes(prob, mdl, P.x, P.phi0);
        fprintf('%-12s alpha=%.2f  %3d it  omega1/2 = %7.2f / %7.2f | phi1: mode %d MAC %.3f %7.1f (%.2fx) | phi2: mode %d MAC %.3f %7.1f (%.2fx)\n', ...
            prob.name, alphas(ia), P.iterations, om, tr(1,1), tr(1,2), tr(1,3), tr(1,3)/omIni(1), ...
            tr(2,1), tr(2,2), tr(2,3), tr(2,3)/omIni(2));
        if ~quick, compare_with_paper(sprintf('%s alpha=%.2f omega1', prob.name, alphas(ia)), om(1), paperOmega1{ip}(ia), 'rad/s'); end
        rows(end+1, :) = {prob.name, alphas(ia), P.iterations, om(1), om(2), info.nRejectedBelow, ...
            tr(1,1), tr(1,2), tr(1,3), tr(1,3)/omIni(1), tr(2,1), tr(2,2), tr(2,3), tr(2,3)/omIni(2)}; %#ok<AGROW>
        save_design_png(prob, P.x, fullfile(out, sprintf('%s_alpha_%.2f.png', prob.name, alphas(ia))));
    end
    fprintf('%s: starting design omega_ini = %.2f / %.2f rad/s\n', prob.name, omIni);
end
T = cell2table(rows, 'VariableNames', {'case', 'alpha', 'iterations', 'omega1', 'omega2', 'n_rejected', ...
    'phi1_mode', 'phi1_MAC', 'phi1_omega', 'phi1_gain', 'phi2_mode', 'phi2_MAC', 'phi2_omega', 'phi2_gain'});
writetable(T, fullfile(out, 'alpha_sweep.csv'));
end

function tr = trackModes(prob, mdl, x, phi0)
% Mode tracking of Section 2.4: among the structural modes of the 100 lowest eigenpairs,
% the best mass-MAC match of each reference shape: [position, MAC, omega].
[~, ~, info] = structural_modes(prob, x, 1, struct('mdl', mdl, 'kStart', 100));
sel = find(info.keep);  f = info.free;
tr = zeros(2, 3);
for j = 1:2
    m = zeros(numel(sel), 1);
    for k = 1:numel(sel), m(k) = mac_mass(phi0(f, j), info.Vraw(:, sel(k)), info.Mf); end
    [best, kb] = max(m);
    tr(j, :) = [kb, best, sqrt(info.lambdaRaw(sel(kb)))];
end
end
