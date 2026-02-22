function [pasS, pasV] = parsePassiveRegions(cfg, nelx, nely, L, H)
%PARSEPASSIVEREGIONS  Convert domain.passive_regions JSON block to element index sets.
%
%   [pasS, pasV] = parsePassiveRegions(cfg, nelx, nely, L, H)
%
% Inputs:
%   cfg        : decoded JSON struct (full config, same as passed to run_topopt_from_json)
%   nelx, nely : mesh dimensions (number of elements in x and y)
%   L, H       : physical domain dimensions
%
% Outputs:
%   pasS : sorted column vector of 1-based element indices forced solid  (density = 1)
%   pasV : sorted column vector of 1-based element indices forced void   (density = 0)
%
% Element numbering (column-major, identical across all three solvers):
%   el = elx*nely + ely + 1   (1-based, elx = 0..nelx-1, ely = 0..nely-1)
%   centroid:  xc = (elx + 0.5)*dx,  yc = (ely + 0.5)*dy
%   where dx = L/nelx, dy = H/nely.
%
% Supported passive_region subtypes (under domain.passive_regions):
%   flanges.top / flanges.bottom  : horizontal strips
%     relative_height in (0,1] : fraction of H clamped at edge
%     is_solid (bool, default true)
%   rect   : one or array of rectangular regions
%     x0   : [xStart, yStart] in physical coordinates
%     size : [width, height], both > 0
%     is_solid (bool, default true)
%
% Overlap policy: solid wins over void.

dx = L / nelx;
dy = H / nely;
nEl = nelx * nely;

% Precompute centroids for all elements (column-major ordering).
elIdx = (0 : nEl-1)';
elx   = floor(elIdx / nely);   % 0-based column (x-direction)
ely   = mod(elIdx, nely);       % 0-based row    (y-direction)
xc    = (elx + 0.5) * dx;
yc    = (ely + 0.5) * dy;

solidMask = false(nEl, 1);
voidMask  = false(nEl, 1);

% No passive_regions field → return empty
if ~isfield(cfg, 'domain') || ~isfield(cfg.domain, 'passive_regions') || ...
        isempty(cfg.domain.passive_regions)
    pasS = zeros(0,1);
    pasV = zeros(0,1);
    return;
end

pr = cfg.domain.passive_regions;

% ---- flanges ----------------------------------------------------------------
if isfield(pr, 'flanges') && ~isempty(pr.flanges)
    fl = pr.flanges;

    if isfield(fl, 'bottom') && ~isempty(fl.bottom)
        b   = fl.bottom;
        rh  = parseRelativeHeight(b, 'domain.passive_regions.flanges.bottom');
        sol = parseIsSolid(b, 'domain.passive_regions.flanges.bottom');
        mask = yc <= H * rh;
        if sol, solidMask = solidMask | mask; else, voidMask = voidMask | mask; end
    end

    if isfield(fl, 'top') && ~isempty(fl.top)
        t   = fl.top;
        rh  = parseRelativeHeight(t, 'domain.passive_regions.flanges.top');
        sol = parseIsSolid(t, 'domain.passive_regions.flanges.top');
        mask = yc >= H * (1 - rh);
        if sol, solidMask = solidMask | mask; else, voidMask = voidMask | mask; end
    end
end

% ---- rect -------------------------------------------------------------------
if isfield(pr, 'rect') && ~isempty(pr.rect)
    rectRaw = pr.rect;
    % jsondecode returns struct array (same fields) or cell array (different fields).
    if isstruct(rectRaw)
        rectList = cell(numel(rectRaw), 1);
        for ki = 1:numel(rectRaw)
            rectList{ki} = rectRaw(ki);
        end
    elseif iscell(rectRaw)
        rectList = rectRaw;
    else
        error('parsePassiveRegions:InvalidRect', ...
            'domain.passive_regions.rect must be a JSON object or array of objects.');
    end

    for k = 1:numel(rectList)
        r     = rectList{k};
        label = sprintf('domain.passive_regions.rect[%d]', k);
        parseRectValidate(r, label);

        x0 = double(r.x0(:));
        sz = double(r.size(:));
        sol = parseIsSolid(r, label);

        % Clip to domain
        xStart = max(0, x0(1));
        yStart = max(0, x0(2));
        xEnd   = min(L, x0(1) + sz(1));
        yEnd   = min(H, x0(2) + sz(2));

        if xEnd <= xStart || yEnd <= yStart
            warning('parsePassiveRegions:EmptyRect', ...
                '%s clips to an empty region — skipped.', label);
            continue;
        end

        mask = (xc >= xStart & xc <= xEnd) & (yc >= yStart & yc <= yEnd);
        if sol, solidMask = solidMask | mask; else, voidMask = voidMask | mask; end
    end
end

% Solid wins over void on overlap.
voidMask = voidMask & ~solidMask;

pasS = find(solidMask);
pasV = find(voidMask);
end

% ---------------------------------------------------------------------------
function rh = parseRelativeHeight(s, label)
if ~isfield(s, 'relative_height') || isempty(s.relative_height)
    error('parsePassiveRegions:MissingField', ...
        '%s: missing required field "relative_height".', label);
end
rh = double(s.relative_height);
if ~isscalar(rh) || ~isfinite(rh) || rh <= 0 || rh > 1
    error('parsePassiveRegions:InvalidRelativeHeight', ...
        '%s.relative_height must be a numeric scalar in (0, 1].', label);
end
end

% ---------------------------------------------------------------------------
function parseRectValidate(r, label)
if ~isfield(r, 'x0') || isempty(r.x0)
    error('parsePassiveRegions:MissingField', '%s: missing required field "x0".', label);
end
x0 = double(r.x0(:));
if numel(x0) ~= 2 || any(~isfinite(x0))
    error('parsePassiveRegions:InvalidRectX0', ...
        '%s.x0 must be a numeric [xStart, yStart] pair.', label);
end
if ~isfield(r, 'size') || isempty(r.size)
    error('parsePassiveRegions:MissingField', '%s: missing required field "size".', label);
end
sz = double(r.size(:));
if numel(sz) ~= 2 || any(~isfinite(sz)) || any(sz <= 0)
    error('parsePassiveRegions:InvalidRectSize', ...
        '%s.size must be a numeric [w, h] pair with w > 0 and h > 0.', label);
end
end

% ---------------------------------------------------------------------------
function b = parseIsSolid(s, label)
if ~isfield(s, 'is_solid') || isempty(s.is_solid)
    b = true;  % default: solid
    return;
end
v = s.is_solid;
if islogical(v) && isscalar(v), b = v; return; end
if isnumeric(v) && isscalar(v), b = (v ~= 0); return; end
if isstring(v) && isscalar(v), v = char(v); end
if ischar(v)
    t = lower(strtrim(v));
    if any(strcmp(t, {'true','yes','1','on'})),  b = true;  return; end
    if any(strcmp(t, {'false','no','0','off'})), b = false; return; end
end
error('parsePassiveRegions:InvalidBoolean', ...
    '%s.is_solid must be boolean-like.', label);
end
