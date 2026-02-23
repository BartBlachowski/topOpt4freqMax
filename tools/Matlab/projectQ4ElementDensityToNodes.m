function [rhoNodal, cache, rebuilt] = projectQ4ElementDensityToNodes(rhoElem, nelx, nely, cache)
%PROJECTQ4ELEMENTDENSITYTONODES Average Q4 element densities onto mesh nodes.
%
%   [rhoNodal, cache] = projectQ4ElementDensityToNodes(rhoElem, nelx, nely)
%   [rhoNodal, cache] = projectQ4ElementDensityToNodes(rhoElem, nelx, nely, cache)
%   [rhoNodal, cache, rebuilt] = projectQ4ElementDensityToNodes(...)
%
% Canonical mapping (column-major in x):
%   nodeId(ex,ey) = (ex-1)*(nely+1) + ey, ex=1..nelx+1, ey=1..nely+1
%   element corners: n1(LL), n2(LR), n3(UR), n4(UL)
%
% Nodal projection is pure averaging of adjacent element values via
% accumarray. This avoids orientation/transposition bias.

if nargin < 4
    cache = [];
end
if ~isscalar(nelx) || ~isscalar(nely) || nelx < 1 || nely < 1 || ...
        nelx ~= round(nelx) || nely ~= round(nely)
    error('projectQ4ElementDensityToNodes:InvalidMeshSize', ...
        'nelx and nely must be positive integers.');
end

nEl = nelx * nely;
nNodes = (nelx + 1) * (nely + 1);
rhoElem = reshape(double(rhoElem), [], 1);
if numel(rhoElem) ~= nEl
    error('projectQ4ElementDensityToNodes:InvalidElementFieldSize', ...
        'rhoElem must contain nelx*nely entries (%d expected, %d provided).', ...
        nEl, numel(rhoElem));
end

rebuilt = false;
if ~localIsCacheValid(cache, nelx, nely, nEl, nNodes)
    cache = localBuildCache(nelx, nely, cache);
    rebuilt = true;
end

% Robust nodal averaging from element values.
vals = repmat(rhoElem, 4, 1);
sumVals = accumarray(cache.elemNodeLinearIdx, vals, [nNodes, 1], @sum, 0);
rhoNodal = sumVals ./ cache.nodeCountsSafe;
end

function ok = localIsCacheValid(cache, nelx, nely, nEl, nNodes)
ok = isstruct(cache) && ...
    isfield(cache, 'version') && isnumeric(cache.version) && isscalar(cache.version) && isfinite(cache.version) && ...
    isfield(cache, 'nelx') && isfield(cache, 'nely') && ...
    isfield(cache, 'nEl') && isfield(cache, 'nNodes') && ...
    isfield(cache, 'elemNodes') && isfield(cache, 'nodeXY') && ...
    isfield(cache, 'elemNodeLinearIdx') && isfield(cache, 'nodeCounts') && ...
    isfield(cache, 'nodeCountsSafe') && isfield(cache, 'Pavg') && ...
    cache.nelx == nelx && cache.nely == nely && ...
    cache.nEl == nEl && cache.nNodes == nNodes && ...
    isequal(size(cache.elemNodes), [nEl, 4]) && ...
    isequal(size(cache.nodeXY), [nNodes, 2]) && ...
    isequal(size(cache.elemNodeLinearIdx), [4*nEl, 1]) && ...
    isequal(size(cache.nodeCounts), [nNodes, 1]) && ...
    isequal(size(cache.nodeCountsSafe), [nNodes, 1]) && ...
    isequal(size(cache.Pavg), [nNodes, nEl]);
end

function cache = localBuildCache(nelx, nely, oldCache)
nEl = nelx * nely;
nNodes = (nelx + 1) * (nely + 1);

% Build Q4 connectivity with canonical node indexing.
nodeNrs = reshape(1:nNodes, nely + 1, nelx + 1);
n1 = nodeNrs(1:nely,   1:nelx);     % LL
n2 = nodeNrs(1:nely,   2:nelx+1);   % LR
n3 = nodeNrs(2:nely+1, 2:nelx+1);   % UR
n4 = nodeNrs(2:nely+1, 1:nelx);     % UL
elemNodes = [n1(:), n2(:), n3(:), n4(:)];

elemNodeLinearIdx = elemNodes(:);
elemLinearIdx = repelem((1:nEl)', 4, 1);
nodeCounts = accumarray(elemNodeLinearIdx, 1, [nNodes, 1], @sum, 0);
nodeCountsSafe = max(nodeCounts, 1);

% Keep the averaging operator for code paths that need Pavg' (semi_harmonic sensitivity).
P = sparse(elemNodeLinearIdx, elemLinearIdx, 1, nNodes, nEl);
Pavg = spdiags(1 ./ nodeCountsSafe, 0, nNodes, nNodes) * P;

% Node coordinates aligned with canonical ids and axis limits [1..nelx+1], [1..nely+1].
[xGrid, yGrid] = meshgrid(1:(nelx + 1), 1:(nely + 1));
nodeXY = [xGrid(:), yGrid(:)];

cache = struct();
cache.version = localNextVersion(oldCache);
cache.nelx = nelx;
cache.nely = nely;
cache.nEl = nEl;
cache.nNodes = nNodes;
cache.elemNodes = elemNodes;
cache.nodeXY = nodeXY;
cache.elemNodeLinearIdx = elemNodeLinearIdx;
cache.nodeCounts = nodeCounts;
cache.nodeCountsSafe = nodeCountsSafe;
cache.Pavg = Pavg;
end

function v = localNextVersion(oldCache)
v = 1;
if isstruct(oldCache) && isfield(oldCache, 'version') && ...
        isnumeric(oldCache.version) && isscalar(oldCache.version) && isfinite(oldCache.version)
    v = oldCache.version + 1;
end
end
