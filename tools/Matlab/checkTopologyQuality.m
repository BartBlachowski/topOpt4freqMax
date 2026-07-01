function check = checkTopologyQuality(x, nelx, nely, varargin)
%CHECKTOPOLOGYQUALITY Morphology guard for revision topology results.
%
%   check = checkTopologyQuality(x, nelx, nely)
%
% The guard is intentionally simple and deterministic.  It rejects highly
% fragmented black/white fields like speckled near-mechanisms even when scalar
% convergence, volume, or mode-tracking checks pass.

p = inputParser;
p.addParameter('solid_threshold', 0.5, @(v) isnumeric(v) && isscalar(v));
p.addParameter('max_solid_components', [], @(v) isempty(v) || (isnumeric(v) && isscalar(v)));
p.addParameter('min_largest_solid_fraction', 0.95, @(v) isnumeric(v) && isscalar(v));
p.addParameter('require_left_right_spanning', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

rho = double(x);
if isvector(rho)
    if numel(rho) ~= nelx * nely
        error('checkTopologyQuality:SizeMismatch', ...
            'Vector topology has %d entries, expected nelx*nely=%d.', ...
            numel(rho), nelx*nely);
    end
    rho = reshape(rho(:), nely, nelx);
else
    if ~isequal(size(rho), [nely, nelx])
        error('checkTopologyQuality:SizeMismatch', ...
            'Matrix topology has size %dx%d, expected %dx%d.', ...
            size(rho,1), size(rho,2), nely, nelx);
    end
end

if isempty(opts.max_solid_components)
    opts.max_solid_components = max(50, ceil(0.0025 * nelx * nely));
end

solid = rho >= opts.solid_threshold;
[~, componentCount, componentSizes, touchesLeft, touchesRight] = ...
    localConnectedComponents4(solid);

solidCells = nnz(solid);
if solidCells > 0 && ~isempty(componentSizes)
    [largestSize, largestIdx] = max(componentSizes);
    largestFraction = largestSize / solidCells;
    largestTouchesLeft = touchesLeft(largestIdx);
    largestTouchesRight = touchesRight(largestIdx);
else
    largestSize = 0;
    largestIdx = 0;
    largestFraction = 0;
    largestTouchesLeft = false;
    largestTouchesRight = false;
end

issues = {};
if solidCells == 0
    issues{end+1} = 'no solid cells above threshold';
end
if componentCount > opts.max_solid_components
    issues{end+1} = sprintf('too many solid components: %d > %d', ...
        componentCount, opts.max_solid_components);
end
if largestFraction < opts.min_largest_solid_fraction
    issues{end+1} = sprintf('largest solid component fraction %.6g < %.6g', ...
        largestFraction, opts.min_largest_solid_fraction);
end
if opts.require_left_right_spanning && ~(largestTouchesLeft && largestTouchesRight)
    issues{end+1} = 'largest solid component does not span left and right supports';
end

check = struct();
check.pass = isempty(issues);
check.issues = issues(:);
check.thresholds = struct( ...
    'solid_threshold', opts.solid_threshold, ...
    'max_solid_components', opts.max_solid_components, ...
    'min_largest_solid_fraction', opts.min_largest_solid_fraction, ...
    'require_left_right_spanning', opts.require_left_right_spanning);
check.metrics = struct( ...
    'solid_cell_count', solidCells, ...
    'solid_fraction', solidCells / numel(solid), ...
    'solid_component_count', componentCount, ...
    'largest_solid_component_id', largestIdx, ...
    'largest_solid_component_size', largestSize, ...
    'largest_solid_component_fraction', largestFraction, ...
    'largest_component_touches_left_support', largestTouchesLeft, ...
    'largest_component_touches_right_support', largestTouchesRight, ...
    'largest_component_touches_both_supports', largestTouchesLeft && largestTouchesRight, ...
    'top_component_sizes', localTopComponentSizes(componentSizes, 10));
end

function [labels, count, sizes, touchesLeft, touchesRight] = localConnectedComponents4(mask)
[nely, nelx] = size(mask);
labels = zeros(nely, nelx);
count = 0;
sizes = zeros(0, 1);
touchesLeft = false(0, 1);
touchesRight = false(0, 1);

queueRows = zeros(numel(mask), 1);
queueCols = zeros(numel(mask), 1);

for col = 1:nelx
    for row = 1:nely
        if ~mask(row, col) || labels(row, col) ~= 0
            continue;
        end

        count = count + 1;
        head = 1;
        tail = 1;
        queueRows(tail) = row;
        queueCols(tail) = col;
        labels(row, col) = count;
        compSize = 0;
        compLeft = false;
        compRight = false;

        while head <= tail
            r = queueRows(head);
            c = queueCols(head);
            head = head + 1;
            compSize = compSize + 1;
            compLeft = compLeft || c == 1;
            compRight = compRight || c == nelx;

            neighbors = [r-1, c; r+1, c; r, c-1; r, c+1];
            for k = 1:4
                rr = neighbors(k, 1);
                cc = neighbors(k, 2);
                if rr < 1 || rr > nely || cc < 1 || cc > nelx
                    continue;
                end
                if mask(rr, cc) && labels(rr, cc) == 0
                    tail = tail + 1;
                    queueRows(tail) = rr;
                    queueCols(tail) = cc;
                    labels(rr, cc) = count;
                end
            end
        end

        sizes(count, 1) = compSize;
        touchesLeft(count, 1) = compLeft;
        touchesRight(count, 1) = compRight;
    end
end
end

function topSizes = localTopComponentSizes(sizes, n)
if isempty(sizes)
    topSizes = zeros(0, 1);
else
    topSizes = sort(sizes(:), 'descend');
    topSizes = topSizes(1:min(n, numel(topSizes)));
end
end
