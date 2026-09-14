function fn = plot_all_topologies(src, fn)
% PLOT_ALL_TOPOLOGIES  Stack the optimized topologies of all cases into one figure.
%
%   fn = plot_all_topologies()
%       Rebuild the panel from the result.mat files already saved under
%       experiments/paper_examples/ by RUN_OLHOFF_CASE -- no re-run needed.
%
%   fn = plot_all_topologies({'ss_n1','cc_n1'})
%       Same, restricted to the named cases and in the order given.
%
%   fn = plot_all_topologies(resCell)
%       Use in-memory result structs (this is how RUN_ALL_OLHOFF_2014 calls it).
%
%   fn = plot_all_topologies(src, fn)
%       Write to an explicit path instead of
%       experiments/paper_examples/topologies_all.png.
%
%   One row per case: the density field, titled with the case name, the paper
%   figure it corresponds to, computed vs paper, multiplicity, the number of
%   structural components and the section 7.1 verdict.

here    = fileparts(mfilename('fullpath'));
outroot = fullfile(here, '..', 'experiments', 'paper_examples');

if nargin < 1 || isempty(src)
    src = {'ss_n1','cs_n1','cc_n1','ss_n2','cs_n2','cc_n2','cc_gap23'};
end
if nargin < 2 || isempty(fn)
    fn = fullfile(outroot, 'topologies_all.png');
end

% ---- Resolve src to a cell array of result structs ----------------------
if isstruct(src)
    res = num2cell(src);
elseif iscell(src) && ~isempty(src) && isstruct(src{1})
    res = src;
else
    if ischar(src) || isstring(src), src = cellstr(src); end
    res = {};
    for k = 1:numel(src)
        f = fullfile(outroot, char(src{k}), 'result.mat');
        if ~exist(f, 'file')
            warning('plot_all_topologies:MissingCase', ...
                'no saved result for case ''%s'' (%s), skipping', char(src{k}), f);
            continue
        end
        S = load(f, 'res');
        res{end+1} = S.res; %#ok<AGROW>
    end
end
if isempty(res)
    error('plot_all_topologies:NoResults', 'nothing to plot');
end

% ---- Layout -------------------------------------------------------------
% Each axes is positioned at the exact aspect of its mesh.  Do NOT lean on
% "axis equal" inside a tiledlayout: the beams are 8:1 and a tile is not, so
% the image gets scaled to the tile height and clipped at the sides.
nc = numel(res);
W = 1000; margin = 25; imgW = W - 2*margin;    % px, nominal
bandH = 46;                                    % two-line title above each axes
gapH  = 18;                                    % breathing room below
topH  = 46;                                    % overall title band

imgH = zeros(1, nc);
for k = 1:nc
    imgH(k) = imgW * res{k}.cfg.nely / res{k}.cfg.nelx;
end
rowH = bandH + imgH + gapH;
H    = topH + sum(rowH);

% ---- Draw ---------------------------------------------------------------
% Positions are normalized and the page aspect is pinned to W:H, so the
% layout survives a figure that the window manager clamps to screen height.
fh = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 W H]);
annotation(fh, 'textbox', [0 1-topH/H 1 topH/H], ...
    'String', 'Olhoff & Du (2014) -- optimized topologies, 2D examples', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 13);

ytop = H - topH;
for k = 1:nc
    r  = res{k};
    y0 = ytop - bandH - imgH(k);
    ax = axes(fh, 'Position', [margin/W, y0/H, imgW/W, imgH(k)/H]);
    imagesc(ax, 1 - reshape(r.rho, r.cfg.nely, r.cfg.nelx));
    colormap(ax, gray); clim(ax, [0 1]); axis(ax, 'off');
    th = title(ax, ...
        {sprintf('%s   (%s)', upper(r.name), r.target.figure), ...
         sprintf('computed %.1f / paper %.1f  (%+.2f %%)   N = %g   comp = %d   %s', ...
                 r.value, r.target_value, r.err_pct, ...
                 r.final_N, r.components, r.verdict)}, ...
        'Interpreter', 'none', 'FontSize', 10);
    set(th, 'Visible', 'on');
    ytop = ytop - rowH(k);
end

d = fileparts(fn);
if ~isempty(d) && ~exist(d, 'dir'), mkdir(d); end
% print, not exportgraphics: it renders from PaperPosition rather than from
% the on-screen size, so the output is identical whatever the display is.
set(fh, 'PaperUnits', 'inches', 'PaperPositionMode', 'manual', ...
        'PaperPosition', [0 0 W H]/100, 'PaperSize', [W H]/100);
print(fh, fn, '-dpng', '-r150');
close(fh);
end
