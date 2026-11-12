function prob = problem_building(nelx, nely, volfrac)
%PROBLEM_BUILDING  Three-story building of Section 4.3.
%   5 m wide, 15 m high, clamped base, E = 1e7 Pa, nu = 0.3, rho = 1 kg/m^3,
%   Emin = 1e-6 E0, rhoMin = 1e-6.  Passive solid frame: side columns 0.1 m wide
%   and floor plates 0.1 m thick at y = 5, 10 and 15 m (elements whose centroid
%   lies in these strips).  V_f = 0.3 of the whole domain, frame included.
%   Defaults: 80 x 240 elements.  Meshes coarser than 40 x 120 cannot resolve the
%   0.1 m frame (element centroids fall outside the strips).
if nargin < 1 || isempty(nelx), nelx = 80; end
if nargin < 2 || isempty(nely), nely = 3*nelx; end
if nargin < 3 || isempty(volfrac), volfrac = 0.3; end
L = 5;  H = 15;
rects = [0 0 0.1 15; 4.9 0 0.1 15; 0 4.9 5 0.1; 0 9.9 5 0.1; 0 14.9 5 0.1];  % [x0 y0 w h]
dx = L/nelx;  dy = H/nely;
e = (0:nelx*nely-1)';
xc = (floor(e/nely) + 0.5)*dx;  yc = (mod(e, nely) + 0.5)*dy;
solid = false(nelx*nely, 1);
for k = 1:size(rects, 1)
    r = rects(k, :);
    xs = max(0, r(1));  ys = max(0, r(2));  xe = min(L, r(1) + r(3));  ye = min(H, r(2) + r(4));
    solid = solid | (xc >= xs & xc <= xe & yc >= ys & yc <= ye);
end
prob = struct('name', 'building', 'L', L, 'H', H, 'nelx', nelx, 'nely', nely, ...
    'E', 1e7, 'nu', 0.3, 'rho', 1, 'volfrac', volfrac, 'support', 'clampedBottom', ...
    'passive', find(solid), 'Emin', 1e-6*1e7, 'rhoMin', 1e-6);
end
