function T = tables3_4_matched_settings(quick, nRep)
%TABLES3_4_MATCHED_SETTINGS  Section 4.1, Tables 3-4: native versus matched settings.
%
%   T = tables3_4_matched_settings()           160x20, 400x50, 800x100, 3 timed runs each
%                                              (about 2 h, Du-Olhoff dominates)
%   T = tables3_4_matched_settings(true)       quick: 160x20, one run
%   T = tables3_4_matched_settings(false, 1)   all meshes, one run
%
%   Native: the settings of each implementation (Table 3).  Matched: the settings of the
%   Du-Olhoff implementation imposed on the two static methods: physical filter radius
%   0.06 m, move limit 0.1, Pedersen stiffness (linear below x = 0.1) with linear mass,
%   floors E = 1e-5 E0 and rho = 1e-3 rho0.  Du-Olhoff itself is unchanged.  Median
%   total wall time of nRep runs after a discarded warm-up; omega_1 under the common
%   evaluator.  Output: results/table4/table4.csv.
if nargin < 1, quick = false; end
if nargin < 2, nRep = 3; end
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('table4');
maxNumCompThreads(1);
meshes = [160 20; 400 50; 800 100];
if quick, meshes = meshes(1, :); nRep = 1; end
matched = struct('filterRadius', 0.06, 'move', 0.1, 'stiffness', 'pedersen', 'Emin', 1e-5*1e7, 'rhoMin', 1e-3);
mY = matched;  mY.massCut = 0;                 % Yuksel-Yilmaz: linear mass under matched settings
w = problem_ss_beam(48, 6);  proposed_freq(w); yuksel_freq(w); du_olhoff(w, struct('maxOuter', 5));   % warm-up
runs = {'proposed', 'native', @(p) proposed_freq(p); 'proposed', 'matched', @(p) proposed_freq(p, matched); ...
        'yuksel', 'native', @(p) yuksel_freq(p); 'yuksel', 'matched', @(p) yuksel_freq(p, mY); ...
        'du_olhoff', 'native=matched', @(p) du_olhoff(p)};
rows = {};
for k = 1:size(meshes, 1)
    prob = problem_ss_beam(meshes(k,1), meshes(k,2));
    for r = 1:size(runs, 1)
        t = zeros(nRep, 1);
        for rep = 1:nRep, S = runs{r,3}(prob); t(rep) = S.time.total; end
        om = structural_modes(prob, S.x, 1);
        fprintf('%4dx%-3d %-9s %-15s median %8.2f s (range %.2f-%.2f)  omega1 %.1f\n', meshes(k,:), runs{r,1}, ...
            runs{r,2}, median(t), min(t), max(t), om);
        rows(end+1, :) = {sprintf('%dx%d', meshes(k,:)), runs{r,1}, runs{r,2}, median(t), min(t), max(t), om}; %#ok<AGROW>
    end
end
T = cell2table(rows, 'VariableNames', {'mesh', 'method', 'settings', 'time_median_s', 'time_min_s', 'time_max_s', 'omega1'});
writetable(T, fullfile(out, 'table4.csv'));
end
