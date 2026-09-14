function xb = exact_count_binary(x, volumeFraction)
%EXACT_COUNT_BINARY Exact solid count; ties use increasing global index.
arguments
    x {mustBeNumeric,mustBeReal,mustBeFinite}
    volumeFraction (1,1) double {mustBeGreaterThanOrEqual(volumeFraction,0),mustBeLessThanOrEqual(volumeFraction,1)}
end
x = double(x(:));
nSolid = round(volumeFraction * numel(x));
[~, order] = sortrows([-x, (1:numel(x)).'], [1 2]);
xb = zeros(size(x));
xb(order(1:nSolid)) = 1;
end
