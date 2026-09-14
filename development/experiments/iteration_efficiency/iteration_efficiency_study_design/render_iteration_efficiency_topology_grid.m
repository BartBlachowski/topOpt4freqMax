function artifact = render_iteration_efficiency_topology_grid(states, outputBase, opts)
%RENDER_ITERATION_EFFICIENCY_TOPOLOGY_GRID Standard accepted-state grid.
%
%   ARTIFACT = RENDER_ITERATION_EFFICIENCY_TOPOLOGY_GRID(STATES,OUTPUTBASE)
%   lays out the states selected by the frozen iteration-efficiency method
%   (k_enter, k_cert, reference, or an explicitly last-observed state). This
%   function selects/layouts states only: every topology cell is drawn by the
%   shared tools/Matlab/renderTopologyDensity.m renderer.
%
%   Required fields in each STATES element:
%     density, nelx, nely, method, status, admissible, state_label
%
%   Optional fields:
%     domain_extent [xmin xmax ymin ymax]
%     representation (for example "raw" or "binary")
%     support_points N-by-2 domain coordinates
%     load_vectors N-by-4 rows [x y fx fy]
%
%   Failed, errored and unavailable records are drawn as labelled empty cells.
%   No earlier density is substituted. Pre-failure diagnostic topologies do
%   not belong in this accepted-state grid and are rejected.
%
%   OUTPUTBASE has no extension. A 300-dpi PNG, vector PDF and MATLAB FIG are
%   written. Plotting/export is presentation-only and must be called after all
%   optimization and endpoint-timing measurements.

if nargin < 3 || isempty(opts); opts = struct(); end
if isempty(states)
    error('render_iteration_efficiency_topology_grid:EmptyStates', ...
        'At least one state record is required.');
end

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo, 'tools', 'Matlab'));

n = numel(states);
gridSize = localOpt(opts, 'GridSize', []);
if isempty(gridSize)
    nCols = min(3, n);
    nRows = ceil(n/nCols);
else
    gridSize = double(gridSize(:).');
    assert(numel(gridSize) == 2 && all(gridSize >= 1) && ...
        prod(gridSize) >= n, ...
        'render_iteration_efficiency_topology_grid:InvalidGridSize', ...
        'GridSize must be [rows columns] with enough cells.');
    nRows = gridSize(1); nCols = gridSize(2);
end

overlayPolicy = char(string(localOpt(opts, 'OverlayPolicy', 'supports')));
figWidth = max(900, 430*nCols);
figHeight = max(300, 245*nRows);
fig = figure('Color', 'white', 'Visible', ...
    char(string(localOpt(opts, 'Visible', 'off'))), ...
    'Units', 'pixels', 'Position', [100 100 figWidth figHeight]);
layout = tiledlayout(fig, nRows, nCols, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

cellInfo = cell(n, 1);
for i = 1:n
    st = states(i);
    localRequireFields(st, {'density','nelx','nely','method','status', ...
        'admissible','state_label'});
    ax = nexttile(layout);

    extent = localField(st, 'domain_extent', [0 st.nelx 0 st.nely]);
    supports = localField(st, 'support_points', zeros(0,2));
    loads = localField(st, 'load_vectors', zeros(0,4));
    representation = char(string(localField(st, 'representation', 'raw')));
    stateLabel = char(string(st.state_label));
    if ~isempty(representation)
        stateLabel = sprintf('%s | %s', stateLabel, representation);
    end

    renderOpts = struct( ...
        'ParentAxes', ax, ...
        'Export', false, ...
        'CloseFigure', false, ...
        'ApproachName', st.method, ...
        'DomainExtent', extent, ...
        'VisualizationQuality', 'regular', ...
        'ResultStatus', st.status, ...
        'Admissible', logical(st.admissible), ...
        'StateKind', 'accepted_state', ...
        'StateLabel', stateLabel, ...
        'OverlayPolicy', overlayPolicy, ...
        'SupportPoints', supports, ...
        'LoadVectors', loads);
    cellInfo{i} = renderTopologyDensity(st.density, st.nelx, st.nely, renderOpts);

    if cellInfo{i}.skipped
        cla(ax);
        axis(ax, 'off');
        text(ax, 0.5, 0.56, char(string(st.method)), ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', ...
            'FontWeight', 'bold', 'Interpreter', 'none');
        text(ax, 0.5, 0.42, sprintf('%s\n%s', char(string(st.status)), ...
            cellInfo{i}.skip_reason), ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', ...
            'Interpreter', 'none', 'Color', [0.55 0.10 0.10]);
    end
end

for i = n+1:nRows*nCols
    ax = nexttile(layout); axis(ax, 'off');
end

[outDir, ~, ~] = fileparts(outputBase);
if ~isempty(outDir) && exist(outDir, 'dir') ~= 7; mkdir(outDir); end
pngPath = [outputBase '.png'];
pdfPath = [outputBase '.pdf'];
figPath = [outputBase '.fig'];
drawnow;
exportgraphics(fig, pngPath, 'Resolution', 300, 'BackgroundColor', 'white');
exportgraphics(fig, pdfPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
savefig(fig, figPath);
close(fig);

artifact = struct();
artifact.png_path = pngPath;
artifact.pdf_path = pdfPath;
artifact.fig_path = figPath;
artifact.cells = cellInfo;
artifact.renderer = 'tools/Matlab/renderTopologyDensity.m';
end

function value = localOpt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name)); value = opts.(name); else; value = default; end
end

function value = localField(s, name, default)
if isfield(s, name) && ~isempty(s.(name)); value = s.(name); else; value = default; end
end

function localRequireFields(s, names)
missing = names(~cellfun(@(name) isfield(s, name), names));
if ~isempty(missing)
    error('render_iteration_efficiency_topology_grid:MissingField', ...
        'State record is missing required field(s): %s.', strjoin(missing, ', '));
end
end
