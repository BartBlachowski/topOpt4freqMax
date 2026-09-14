function img = topologyImage(rho, nelx, nely, pxPerEl)
%TOPOLOGYIMAGE  Render a density field the way the paper prints it:
%   black = solid (rho = 1), white = void (rho = 0).
if nargin < 4, pxPerEl = 6; end
X = reshape(rho(:), nely, nelx);          % column-major, matches model2D
G = uint8(255*(1 - min(max(X,0),1)));
img = repelem(G, pxPerEl, pxPerEl);
end
