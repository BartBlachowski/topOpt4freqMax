function xt = a4_volume_preserving_threshold(x, volfrac, materialFloor)
%A4_VOLUME_PRESERVING_THRESHOLD  Binary threshold using the configured floor.
%
%   Solid elements are assigned 1. Void elements are assigned the actual
%   configured material floor supplied by the endpoint context (rho_min).
%   No undeclared numerical floor is introduced here.

x = double(x(:));
materialFloor = double(materialFloor);
if ~isscalar(materialFloor) || ~isfinite(materialFloor) || ...
        materialFloor < 0 || materialFloor > 1
    error('a4_volume_preserving_threshold:InvalidFloor', ...
        'materialFloor must be a finite scalar in [0,1].');
end
if ~isscalar(volfrac) || ~isfinite(volfrac) || volfrac <= 0 || volfrac > 1
    error('a4_volume_preserving_threshold:InvalidVolumeFraction', ...
        'volfrac must be a finite scalar in (0,1].');
end

xs = sort(x, 'descend');
nKeep = max(1, min(numel(x), round(volfrac * numel(x))));
thr = xs(nKeep);
xt = ones(size(x));
xt(x < thr) = materialFloor;
end
