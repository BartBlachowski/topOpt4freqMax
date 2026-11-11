function c6_make_figures(outDir)
%C6_MAKE_FIGURES  AppliSci revision C6: topology figures for Figures 10-11.
%
%   Renders the ten alpha-sweep designs (results/<case>/alpha_*/result.mat)
%   with the repository renderer (renderTopologyDensity, smooth quality) and
%   titles carrying the STRUCTURAL omega1/omega2 of results/<case>/summary.csv.
%   Output names follow the manuscript: alpha_1..5.png (clamped beam) and
%   building_alpha_1..5.png (building), index 1..5 = alpha 1, 0.75, 0.5, 0.25, 0.
%   Each PNG is then cropped to the design domain (title and margins removed;
%   the frequencies go in the caption), so main.tex includes it without trim.
here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
if nargin < 1 || isempty(outDir), outDir = fullfile(here, 'results', 'figures'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
tags = {'1p00', '0p75', '0p50', '0p25', '0p00'};
for c = {'clamped', 'building'}
    name = c{1};
    T = readtable(fullfile(here, 'results', name, 'summary.csv'));
    for k = 1:5
        S = load(fullfile(here, 'results', name, ['alpha_' tags{k}], 'result.mat'), 'x', 'cfg', 'a');
        row = abs(T.alpha - S.a) < 1e-9;
        nelx = S.cfg.domain.mesh.nelx; nely = S.cfg.domain.mesh.nely;
        if strcmp(name, 'clamped'), base = sprintf('alpha_%d', k); else, base = sprintf('building_alpha_%d', k); end
        renderTopologyDensity(S.x(:), nelx, nely, struct( ...
            'ApproachName', sprintf('alpha = %.2f', S.a), 'OutputDir', outDir, 'OutputBase', base, ...
            'Omega1', T.omega1(row), 'Omega2', T.omega2(row), 'VisualizationQuality', 'smooth', ...
            'DomainExtent', [0 S.cfg.domain.size.length 0 S.cfg.domain.size.height], ...
            'Visible', 'off', 'ResultStatus', 'converged', 'Admissible', true, 'StateKind', 'final'));
        cropTitle(fullfile(outDir, [base '.png']));
    end
end
end

function cropTitle(file)
% Remove the title band and the white margins of an exported topology PNG.
A = imread(file);
G = A; if size(G, 3) == 3, G = rgb2gray(G); end
dark = any(G < 128, 2);
i = find(dark, 1);                       % title starts
i = i - 1 + find(~dark(i:end), 1);       % title ends
i = i - 1 + find(dark(i:end), 1);        % domain starts
B = G(i:end, :);
r = find(any(B < 250, 2)); c = find(any(B < 250, 1));
A = A(i - 1 + (r(1):r(end)), c(1):c(end), :);
imwrite(A, file);
end
