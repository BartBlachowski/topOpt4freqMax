function c8_make_figure(outDir)
%C8_MAKE_FIGURE  AppliSci revision C8: building designs of the proposed method and
%   of the direct (Du-Olhoff) optimization, alpha = 1, for Section 4.4.
%   Reads results/c8c9/c8c9_building_80x240_vf30.mat (xP, xO in the numbering of
%   the alpha sweep, omega under the common evaluator) and renders both as grey-scale
%   density maps: building_direct_proposed.png and
%   building_direct_olhoff.png.
here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
if nargin < 1 || isempty(outDir), outDir = fullfile(here, 'results', 'figures'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
S = load(fullfile(here, 'results', 'c8c9', 'c8c9_building_80x240_vf30.mat'), 'xP', 'xO', 'omP', 'omO');
xs = {S.xP, S.xO}; om = {S.omP, S.omO}; base = {'building_direct_proposed', 'building_direct_olhoff'};
for k = 1:2
    % plain grey-scale density (white = 0, black = 1): the smooth renderer of
    % c6_make_figures thresholds the field and hides the grey members of the
    % Du-Olhoff design, which the comparison is about
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'pixels', 'Position', [100 100 400 1200]);
    ax = axes(fig, 'Position', [0 0 1 1]);
    imagesc(ax, [0 5], [0 15], 1 - reshape(xs{k}(:), 240, 80)); set(ax, 'YDir', 'normal');
    colormap(ax, gray); caxis(ax, [0 1]); axis(ax, 'image', 'off');
    exportgraphics(fig, fullfile(outDir, [base{k} '.png']), 'Resolution', 300); close(fig);
    fprintf('%s: omega1/2 = %.2f / %.2f rad/s, grey = %.3f\n', base{k}, om{k}(1), om{k}(2), 4*mean(xs{k}.*(1-xs{k})));
end
% clamped beam, alpha = 1 (audit 2.9)
S = load(fullfile(here, 'results', 'c8c9', 'c8c9_clamped_400x50_vf50.mat'), 'xP', 'xO', 'omP', 'omO');
xs = {S.xP, S.xO}; om = {S.omP, S.omO}; base = {'clamped_direct_proposed', 'clamped_direct_olhoff'};
for k = 1:2
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'pixels', 'Position', [100 100 1600 200]);
    ax = axes(fig, 'Position', [0 0 1 1]);
    imagesc(ax, [0 8], [0 1], 1 - reshape(xs{k}(:), 50, 400)); set(ax, 'YDir', 'normal');
    colormap(ax, gray); caxis(ax, [0 1]); axis(ax, 'image', 'off');
    exportgraphics(fig, fullfile(outDir, [base{k} '.png']), 'Resolution', 300); close(fig);
    fprintf('%s: omega1/2 = %.2f / %.2f rad/s, grey = %.3f\n', base{k}, om{k}(1), om{k}(2), 4*mean(xs{k}.*(1-xs{k})));
end
end

function cropTitle(file)
% as in c6_make_figures
A = imread(file);
G = A; if size(G, 3) == 3, G = rgb2gray(G); end
dark = any(G < 128, 2);
i = find(dark, 1);
i = i - 1 + find(~dark(i:end), 1);
i = i - 1 + find(dark(i:end), 1);
B = G(i:end, :);
r = find(any(B < 250, 2)); c = find(any(B < 250, 1));
A = A(i - 1 + (r(1):r(end)), c(1):c(end), :);
imwrite(A, file);
end
