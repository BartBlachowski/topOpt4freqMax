function info = renderTopologyDensity(x, nelx, nely, opts)
%RENDERTOPOLOGYDENSITY Shared final/accepted-state topology renderer.
%
%   INFO = RENDERTOPOLOGYDENSITY(X,NELX,NELY,OPTS) renders an element-density
%   field with the presentation convention historically used by
%   RUN_TOPOPT_FROM_JSON for Proposed and Yuksel final snapshots:
%
%     * element rows are shown bottom-to-top (YDir = normal),
%     * solid is black and void is white on a fixed [0,1] density scale,
%     * the complete rectangular domain is shown with equal data units,
%     * support/load overlays are absent unless explicitly requested, and
%     * standalone files use <approach>_<nelx>x<nely>.png/.fig.
%
%   This function is presentation-only. It performs no solve, evaluator, or
%   timing operation. Callers must invoke it outside all measured regions.
%
%   Important OPTS fields (all optional):
%     ApproachName          title/file method name (default "topopt")
%     OutputDir             export directory (default pwd)
%     OutputBase            explicit basename without extension
%     Omega1, Omega2        optional title frequencies [rad/s]
%     VisualizationQuality  "regular" or "smooth" (default "regular")
%     DomainExtent          optional [xmin xmax ymin ymax]; omitted preserves
%                           the legacy element-index imagesc/axis-tight extent
%     FigurePosition        optional pixel position/size; omitted preserves
%                           MATLAB's default Proposed/Yuksel figure size
%     ParentAxes            render into an existing grid axes
%     Export                export standalone PNG/FIG (default true)
%     CloseFigure           close a created figure (default true)
%     Visible               standalone visibility "on"/"off" (default "on")
%     ResultStatus          status label used by the safety gate
%     Admissible            true only for a reportable state (default true)
%     StateKind             "final", "accepted_state", or
%                           "diagnostic_pre_failure"
%     StateLabel            extra title label (k_enter, reference state, ...)
%     OverlayPolicy         "none", "supports", or "supports_and_loads"
%     SupportPoints         N-by-2 normalized support marker coordinates
%     LoadVectors           N-by-4 rows [x y fx fy]
%
%   Final and accepted-state rendering is refused for inadmissible, failed,
%   errored, unavailable, or not-run records. A diagnostic pre-failure image
%   must use StateKind="diagnostic_pre_failure"; it is labelled explicitly
%   and receives a separate filename suffix.

if nargin < 4 || isempty(opts)
    opts = struct();
end

validateattributes(nelx, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(nely, {'numeric'}, {'scalar','integer','positive','finite'});

info = localEmptyInfo();
status = char(string(localOpt(opts, 'ResultStatus', '')));
admissible = logical(localOpt(opts, 'Admissible', true));
stateKind = lower(strtrim(char(string(localOpt(opts, 'StateKind', 'final')))));
stateLabel = char(string(localOpt(opts, 'StateLabel', '')));

validKinds = {'final','accepted_state','diagnostic_pre_failure'};
if ~any(strcmp(stateKind, validKinds))
    error('renderTopologyDensity:InvalidStateKind', ...
        'StateKind must be final, accepted_state, or diagnostic_pre_failure.');
end

if ~strcmp(stateKind, 'diagnostic_pre_failure') && ...
        (~admissible || localStatusForbidsResult(status))
    info.skipped = true;
    info.skip_reason = sprintf('No final/accepted topology for status "%s".', status);
    info.result_status = status;
    return;
end
if strcmp(stateKind, 'diagnostic_pre_failure') && isempty(strtrim(stateLabel))
    stateLabel = 'PRE-FAILURE DIAGNOSTIC';
end

x = double(x(:));
if numel(x) ~= nelx*nely || any(~isfinite(x))
    if strcmp(stateKind, 'diagnostic_pre_failure')
        error('renderTopologyDensity:InvalidDiagnosticField', ...
            'A diagnostic topology must be a finite nelx*nely density field.');
    end
    info.skipped = true;
    info.skip_reason = 'Final/accepted density field is unavailable or nonfinite.';
    info.result_status = status;
    return;
end

approachName = char(string(localOpt(opts, 'ApproachName', 'topopt')));
outputDir = char(string(localOpt(opts, 'OutputDir', pwd)));
outputBase = char(string(localOpt(opts, 'OutputBase', '')));
omega1 = double(localOpt(opts, 'Omega1', NaN));
omega2 = double(localOpt(opts, 'Omega2', NaN));
quality = char(string(localOpt(opts, 'VisualizationQuality', 'regular')));
domainExtent = double(localOpt(opts, 'DomainExtent', []));
figurePosition = double(localOpt(opts, 'FigurePosition', []));
parentAxes = localOpt(opts, 'ParentAxes', []);
doExport = logical(localOpt(opts, 'Export', isempty(parentAxes)));
closeFigure = logical(localOpt(opts, 'CloseFigure', isempty(parentAxes)));
visible = char(string(localOpt(opts, 'Visible', 'on')));
overlayPolicy = lower(strtrim(char(string(localOpt(opts, 'OverlayPolicy', 'none')))));
supportPoints = double(localOpt(opts, 'SupportPoints', zeros(0,2)));
loadVectors = double(localOpt(opts, 'LoadVectors', zeros(0,4)));

if ~isempty(domainExtent)
    validateattributes(domainExtent, {'numeric'}, {'vector','numel',4,'finite'});
    assert(domainExtent(2) > domainExtent(1) && domainExtent(4) > domainExtent(3), ...
        'renderTopologyDensity:InvalidDomainExtent', ...
        'DomainExtent must be [xmin xmax ymin ymax] with positive spans.');
end
if ~isempty(figurePosition)
    validateattributes(figurePosition, {'numeric'}, {'vector','numel',4,'finite'});
    assert(all(figurePosition(3:4) > 0), 'renderTopologyDensity:InvalidFigurePosition', ...
        'FigurePosition width and height must be positive.');
end
if ~any(strcmp(overlayPolicy, {'none','supports','supports_and_loads'}))
    error('renderTopologyDensity:InvalidOverlayPolicy', ...
        'OverlayPolicy must be none, supports, or supports_and_loads.');
end
if ~isempty(supportPoints) && size(supportPoints,2) ~= 2
    error('renderTopologyDensity:InvalidSupportPoints', 'SupportPoints must be N-by-2.');
end
if ~isempty(loadVectors) && size(loadVectors,2) ~= 4
    error('renderTopologyDensity:InvalidLoadVectors', 'LoadVectors must be N-by-4.');
end

createdFigure = isempty(parentAxes);
if createdFigure
    fig = figure('Color', 'white', 'Visible', visible);
    if ~isempty(figurePosition)
        set(fig, 'Units', 'pixels', 'Position', figurePosition);
    end
    try
        theme('light');
    catch
        % theme() is cosmetic and unavailable in older MATLAB releases.
    end
    ax = axes('Parent', fig);
else
    if ~isgraphics(parentAxes, 'axes')
        error('renderTopologyDensity:InvalidParentAxes', ...
            'ParentAxes must be a valid axes handle.');
    end
    ax = parentAxes;
    fig = ancestor(ax, 'figure');
    cla(ax);
end

img = buildTopologyDisplayImage(x, nelx, nely, quality, true);
if isempty(domainExtent)
    imageHandle = imagesc(ax, 1-img, 'Interpolation', 'nearest');
    axis(ax, 'equal');
    axis(ax, 'tight');
    domainExtent = [get(ax, 'XLim') get(ax, 'YLim')];
else
    imageHandle = imagesc(ax, domainExtent(1:2), domainExtent(3:4), 1-img, ...
        'Interpolation', 'nearest');
    axis(ax, 'equal');
    set(ax, 'XLim', domainExtent(1:2), 'YLim', domainExtent(3:4));
end
set(ax, 'YDir', 'normal', 'XColor', 'none', 'YColor', 'none');
colormap(ax, gray(256));
clim(ax, [0 1]);

holdState = ishold(ax);
if ~strcmp(overlayPolicy, 'none')
    hold(ax, 'on');
    if ~isempty(supportPoints)
        plot(ax, supportPoints(:,1), supportPoints(:,2), '^', ...
            'LineStyle', 'none', 'MarkerSize', 7, 'LineWidth', 1.0, ...
            'MarkerEdgeColor', [0.10 0.25 0.80], 'MarkerFaceColor', 'white', ...
            'Tag', 'TopologySupportOverlay');
    end
end
if strcmp(overlayPolicy, 'supports_and_loads') && ~isempty(loadVectors)
    span = [domainExtent(2)-domainExtent(1), domainExtent(4)-domainExtent(3)];
    lengths = hypot(loadVectors(:,3), loadVectors(:,4));
    scale = 0.10 * min(span) / max([lengths; eps]);
    quiver(ax, loadVectors(:,1), loadVectors(:,2), ...
        scale*loadVectors(:,3), scale*loadVectors(:,4), 0, ...
        'Color', [0.80 0.10 0.10], 'LineWidth', 1.2, ...
        'MaxHeadSize', 0.8, 'Tag', 'TopologyLoadOverlay');
end
if ~holdState
    hold(ax, 'off');
end
% Overlay primitives must never expand the standardized domain.
set(ax, 'XLim', domainExtent(1:2), 'YLim', domainExtent(3:4));

nameDisplay = strrep(approachName, '_', ' ');
if isfinite(omega1)
    titleStr = sprintf('%s  |  %dx%d  |  \\omega_{1} = %.2f rad/s', ...
        nameDisplay, nelx, nely, omega1);
    if isfinite(omega2)
        titleStr = sprintf('%s  |  \\omega_{2} = %.2f rad/s', titleStr, omega2);
    end
else
    titleStr = sprintf('%s  |  %dx%d', nameDisplay, nelx, nely);
end
if strcmp(stateKind, 'diagnostic_pre_failure')
    titleStr = sprintf('%s  |  PRE-FAILURE DIAGNOSTIC', titleStr);
end
if ~isempty(strtrim(stateLabel)) && ...
        ~(strcmp(stateKind, 'diagnostic_pre_failure') && ...
          strcmpi(strtrim(stateLabel), 'PRE-FAILURE DIAGNOSTIC'))
    titleStr = sprintf('%s  |  %s', titleStr, stateLabel);
end
title(ax, titleStr, 'Interpreter', 'tex', 'FontSize', 10);

drawnow;

nameSafe = regexprep(approachName, '[^\w\-]', '_');
if isempty(outputBase)
    outputBase = sprintf('%s_%dx%d', nameSafe, nelx, nely);
end
if strcmp(stateKind, 'diagnostic_pre_failure') && ...
        ~endsWith(lower(outputBase), '_pre_failure_diagnostic')
    outputBase = [outputBase '_pre_failure_diagnostic'];
end
basePath = fullfile(outputDir, outputBase);

pngPath = '';
figPath = '';
if doExport
    if exist(outputDir, 'dir') ~= 7
        mkdir(outputDir);
    end
    pngPath = [basePath '.png'];
    figPath = [basePath '.fig'];
    exportgraphics(fig, pngPath, 'Resolution', 160, 'BackgroundColor', 'white');
    savefig(fig, figPath);
    fprintf('Saved topology image: %s  (.png / .fig)\n', basePath);
end

info.skipped = false;
info.skip_reason = '';
info.result_status = status;
info.state_kind = stateKind;
info.figure = fig;
info.axes = ax;
info.image = imageHandle;
info.base_path = basePath;
info.png_path = pngPath;
info.fig_path = figPath;
info.geometry = struct( ...
    'x_lim', get(ax, 'XLim'), ...
    'y_lim', get(ax, 'YLim'), ...
    'y_dir', get(ax, 'YDir'), ...
    'clim', clim(ax), ...
    'data_aspect_ratio', get(ax, 'DataAspectRatio'), ...
    'figure_size_px', localFigureSize(fig), ...
    'display_image_size', size(img), ...
    'overlay_policy', overlayPolicy);

if createdFigure && closeFigure
    close(fig);
    info.figure = [];
    info.axes = [];
    info.image = [];
end
end

function value = localOpt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    value = opts.(name);
else
    value = default;
end
end

function tf = localStatusForbidsResult(status)
key = upper(strtrim(status));
if isempty(key)
    tf = false;
    return;
end
forbidden = {'SOLVER_FAILURE','RUN_ERROR','UNAVAILABLE','NOT_RUN', ...
    'UNVERIFIABLE_AT_PRESENT','N/A'};
parts = regexp(key, '[|,; ]+', 'split');
tf = any(ismember(parts, forbidden)) || contains(key, 'UNAVAILABLE');
end

function sz = localFigureSize(fig)
oldUnits = get(fig, 'Units');
set(fig, 'Units', 'pixels');
pos = get(fig, 'Position');
set(fig, 'Units', oldUnits);
sz = pos(3:4);
end

function info = localEmptyInfo()
info = struct('skipped', false, 'skip_reason', '', 'result_status', '', ...
    'state_kind', '', 'figure', [], 'axes', [], 'image', [], ...
    'base_path', '', 'png_path', '', 'fig_path', '', 'geometry', struct());
end
