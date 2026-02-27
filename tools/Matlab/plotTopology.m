function plotTopology(xPhys, nelx, nely, titleStr, doDrawNow, visualizationQuality, isFinalPlot)
%PLOTTOPOLOGY Plot/update topology in a single reusable figure.
%
%   plotTopology(xPhys, nelx, nely, titleStr, doDrawNow, visualizationQuality, isFinalPlot)
%
% visualizationQuality: 'regular' | 'smooth' (kept for API compatibility)
% isFinalPlot          : kept for API compatibility
%
% Stripe-artifact root cause:
%   Debug checks showed raw xPhys/rhoNodal matrices do not contain barcode
%   columns while rendered frames could. The prior center-subdivided patch
%   also amplified seam patterns on slender elements. This version uses:
%   1) canonical Q4 connectivity + robust nodal projection cache
%   2) shared-node triangulation (2 triangles/quad, alternating diagonal)
%   3) persistent patch handle; update only FaceVertexCData per iteration
%
% Debug mode (default off):
%   Set debugPlot=true below, rerun, and inspect MAT files in tempdir.
%   Files include xPhys_mat, rhoNodal, rhoCenter, rhoVertexPlot, and a
%   painters-renderer ablation metric to separate data issues vs renderer.

persistent figHandle axHandle patchHandle projectionCache meshCache plotCallCount

debugPlot = false;            % set true manually when investigating artifacts
debugStride = 50;             % save on 1st call and every N calls
forcePaintersFallback = false; % leave false unless renderer is proven cause

if nargin < 5 || isempty(doDrawNow)
    doDrawNow = false;
end
if nargin < 6 || isempty(visualizationQuality)
    visualizationQuality = 'regular';
end
if nargin < 7 || isempty(isFinalPlot)
    isFinalPlot = false;
end

% Keep legacy options accepted; renderer is shared between methods.
localParseVisualizationQuality(visualizationQuality); %#ok<NASGU>
if ~islogical(isFinalPlot) && ~(isnumeric(isFinalPlot) && isscalar(isFinalPlot))
    error('plotTopology:InvalidFinalFlag', 'isFinalPlot must be boolean-like.');
end

nEl = nelx * nely;
nNodes = (nelx + 1) * (nely + 1);
xPhys = reshape(double(xPhys), [], 1);
if numel(xPhys) ~= nEl
    error('plotTopology:InvalidFieldSize', ...
        'xPhys must contain nelx*nely entries (%d expected, %d provided).', ...
        nEl, numel(xPhys));
end

[rhoNodal, projectionCache, projectionRebuilt] = projectQ4ElementDensityToNodes( ...
    xPhys, nelx, nely, projectionCache);

if projectionRebuilt || ~localIsMeshCacheValid(meshCache, projectionCache, nEl, nNodes)
    meshCache = localBuildTriMesh(projectionCache);
    patchHandle = [];
end

% Preserve previous visual convention: black = solid, white = void.
rhoVertexPlot = 1 - rhoNodal;

if isempty(plotCallCount) || ~isscalar(plotCallCount) || ~isfinite(plotCallCount)
    plotCallCount = 0;
end
plotCallCount = plotCallCount + 1;

if isempty(figHandle) || ~isgraphics(figHandle, 'figure')
    figHandle = figure('Name', 'Topology', 'NumberTitle', 'off');
    theme("light");
    if isprop(figHandle, 'GraphicsSmoothing')
        set(figHandle, 'GraphicsSmoothing', 'off');
    end
    axHandle = axes('Parent', figHandle);
    patchHandle = [];
end
if isempty(axHandle) || ~isgraphics(axHandle, 'axes')
    axHandle = axes('Parent', figHandle);
    patchHandle = [];
end

if forcePaintersFallback && ~strcmpi(get(figHandle, 'Renderer'), 'painters')
    set(figHandle, 'Renderer', 'painters');
end

if isempty(patchHandle) || ~isgraphics(patchHandle, 'patch')
    patchHandle = patch( ...
        'Parent', axHandle, ...
        'Faces', meshCache.faces, ...
        'Vertices', meshCache.vertices, ...
        'FaceVertexCData', rhoVertexPlot, ...
        'FaceColor', 'interp', ...
        'EdgeColor', 'none', ...
        'LineStyle', 'none');
else
    set(patchHandle, 'FaceVertexCData', rhoVertexPlot);
end

axis(axHandle, 'equal');
axis(axHandle, 'off');
set(axHandle, 'YDir', 'normal');
xlim(axHandle, [1, nelx + 1]);
ylim(axHandle, [1, nely + 1]);
colormap(axHandle, gray(256));
caxis(axHandle, [0 1]);
title(axHandle, titleStr, 'Interpreter', 'none');

if debugPlot
    rhoCenter = mean(rhoNodal(projectionCache.elemNodes), 2);
    localDebugDump(plotCallCount, debugStride, xPhys, nelx, nely, rhoNodal, ...
        rhoCenter, rhoVertexPlot, projectionCache, figHandle, axHandle);
end

if doDrawNow
    drawnow;
end
end

function tf = localIsMeshCacheValid(meshCache, projectionCache, nEl, nNodes)
tf = isstruct(meshCache) && ...
    isfield(meshCache, 'projectionVersion') && ...
    isfield(meshCache, 'nEl') && isfield(meshCache, 'nNodes') && ...
    isfield(meshCache, 'faces') && isfield(meshCache, 'vertices') && ...
    meshCache.projectionVersion == projectionCache.version && ...
    meshCache.nEl == nEl && meshCache.nNodes == nNodes && ...
    isequal(size(meshCache.faces), [2*nEl, 3]) && ...
    isequal(size(meshCache.vertices), [nNodes, 2]);
end

function mesh = localBuildTriMesh(projectionCache)
nEl = projectionCache.nEl;
nely = projectionCache.nely;
elemNodes = projectionCache.elemNodes;

vertices = projectionCache.nodeXY;
faces = zeros(2 * nEl, 3);
for e = 1:nEl
    idx = 2 * (e - 1) + (1:2);
    n1 = elemNodes(e, 1);
    n2 = elemNodes(e, 2);
    n3 = elemNodes(e, 3);
    n4 = elemNodes(e, 4);
    ex = floor((e - 1) / nely);
    ey = mod(e - 1, nely);
    if mod(ex + ey, 2) == 0
        faces(idx, :) = [n1, n2, n3; ...
                         n1, n3, n4];
    else
        faces(idx, :) = [n1, n2, n4; ...
                         n2, n3, n4];
    end
end

mesh = struct();
mesh.projectionVersion = projectionCache.version;
mesh.nEl = nEl;
mesh.nNodes = projectionCache.nNodes;
mesh.faces = faces;
mesh.vertices = vertices;
end

function localDebugDump(it, stride, xPhys, nelx, nely, rhoNodal, rhoCenter, rhoVertexPlot, projectionCache, figHandle, axHandle)
if ~(it == 1 || mod(it, stride) == 0)
    return;
end

xPhys_mat = reshape(xPhys, [nely, nelx]); %#ok<NASGU>
xPhys_mat_T = xPhys_mat.'; %#ok<NASGU>
elemNodesHead = projectionCache.elemNodes(1:min(12, size(projectionCache.elemNodes, 1)), :); %#ok<NASGU>
nodeXYHead = projectionCache.nodeXY(1:min(20, size(projectionCache.nodeXY, 1)), :); %#ok<NASGU>

rendererBefore = '';
rendererAblationDiff = NaN;
if ~isempty(figHandle) && isgraphics(figHandle, 'figure')
    rendererBefore = get(figHandle, 'Renderer');
    % Renderer ablation check: if stripe pattern persists under painters,
    % the issue is in data/connectivity/caching, not OpenGL sampling.
    try
        drawnow;
        frameA = getframe(axHandle);
        set(figHandle, 'Renderer', 'painters');
        drawnow;
        frameB = getframe(axHandle);
        if isequal(size(frameA.cdata), size(frameB.cdata))
            rendererAblationDiff = mean(abs(double(frameA.cdata(:)) - double(frameB.cdata(:))));
        end
        set(figHandle, 'Renderer', rendererBefore);
        drawnow;
    catch
        if ~isempty(rendererBefore)
            try
                set(figHandle, 'Renderer', rendererBefore);
            catch
            end
        end
    end
end

timestamp = datestr(now, 'yyyymmdd_HHMMSS_FFF');
outPath = fullfile(tempdir, sprintf('topology_plot_debug_%s_it%05d.mat', timestamp, it));
save(outPath, ...
    'it', 'nelx', 'nely', ...
    'xPhys_mat', 'xPhys_mat_T', ...
    'rhoNodal', 'rhoCenter', 'rhoVertexPlot', ...
    'elemNodesHead', 'nodeXYHead', ...
    'rendererBefore', 'rendererAblationDiff');
fprintf('[plotTopology debug] Saved %s\n', outPath);
end

function quality = localParseVisualizationQuality(value)
if isstring(value) && isscalar(value)
    value = char(value);
end
if ischar(value)
    key = lower(strtrim(value));
    if isempty(key)
        quality = 'regular';
        return;
    end
    if any(strcmp(key, {'regular', 'smooth'}))
        quality = key;
        return;
    end
end
error('plotTopology:InvalidVisualizationQuality', ...
    'visualization_quality must be "regular" or "smooth".');
end
