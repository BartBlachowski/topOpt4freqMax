function D = fi_directions(S)
%FI_DIRECTIONS  The preregistered deterministic direction set (sec. 6).
%
%   Every direction is unit-2-norm and ADMISSIBLE: at the largest preregistered
%   step (1e-3) the perturbed density stays inside [rhomin, 1].  Elements that
%   would leave the box are zeroed and the direction renormalized; the count is
%   recorded, as the preregistration requires.

rho = S.rho386(:);
nelx = S.nelx; nely = S.nely; NE = S.NE;
R = double(rho > 0.1 & rho < 0.9);
G = reshape(rho, nely, nelx);                 % column-major, matches model2D
Gmask = reshape(R, nely, nelx);

% --- broad gray core: gray elements deeper than the filter radius ---------
depth = bwdist(~Gmask);                       % distance to nearest non-gray
core  = depth >= S.rminEl;
D.core_fraction = mean(core(:));
D.core_maxdepth = max(depth(:));

% --- interface: maximal |grad rho| ---------------------------------------
[gx, gy] = gradient(G);
gm = sqrt(gx.^2 + gy.^2);
[~, iMax] = max(gm(:));
[iy, ix] = ind2sub([nely nelx], iMax);

% --- two disjoint core bumps: deepest core element in each half -----------
half = false(nely, nelx); half(:,1:floor(nelx/2)) = true;
dL = depth; dL(~(core & half))  = -inf; [~, kL] = max(dL(:)); [ly, lx] = ind2sub([nely nelx], kL);
dR = depth; dR(~(core & ~half)) = -inf; [~, kR] = max(dR(:)); [ry, rx] = ind2sub([nely nelx], kR);

bump = @(cx, cy, w) local_bump(nelx, nely, cx, cy, w);
[XX, ~] = meshgrid(1:nelx, 1:nely);
L = nelx;

raw = struct();
raw.D1a = cos(pi*(XX-0.5)/L);
raw.D1b = cos(2*pi*(XX-0.5)/L);
raw.D2a = bump(lx, ly, S.rminEl);
raw.D2b = bump(rx, ry, S.rminEl);
raw.D3  = bump(ix, iy, S.rminEl);
rng(0,'twister'); raw.D4a = randn(nely, nelx);
rng(1,'twister'); raw.D4b = randn(nely, nelx);

names = fieldnames(raw);
D.info = struct();
for k = 1:numel(names)
    v = raw.(names{k})(:);
    [v, nz] = local_admissible(v, rho, S.rhomin, 1e-3);
    D.(names{k}) = v;
    D.info.(names{k}) = struct('zeroed', nz, 'maxabs', max(abs(v)), ...
                               'sum', sum(v), 'norm', norm(v));
end
% --- D5: volume-neutral variants -----------------------------------------
for nm = {'D1a','D1b','D2a','D4a'}
    v = D.(nm{1});
    v = v - mean(v);
    [v, nz] = local_admissible(v, rho, S.rhomin, 1e-3);
    key = ['D5' nm{1}];
    D.(key) = v;
    D.info.(key) = struct('zeroed', nz, 'maxabs', max(abs(v)), ...
                          'sum', sum(v), 'norm', norm(v));
end

D.pairs = { 'D1a','D1b'; 'D2a','D2b'; 'D1a','D2a'; 'D3','D2a'; 'D4a','D4b'; ...
            'D1a','D4a'; 'D2a','D4a'; 'D5D1a','D5D1b'; 'D5D2a','D5D4a'; 'D3','D4b' };
% D5D1b is needed by pair 8
v = D.D1b - mean(D.D1b);
[v, nz] = local_admissible(v, rho, S.rhomin, 1e-3);
D.D5D1b = v;
D.info.D5D1b = struct('zeroed', nz, 'maxabs', max(abs(v)), 'sum', sum(v), 'norm', norm(v));

D.core_mask = core(:);
D.gray_mask = R > 0;
D.depth = depth(:);
D.interface_element = sub2ind([nely nelx], iy, ix);
D.core_elements = [sub2ind([nely nelx], ly, lx), sub2ind([nely nelx], ry, rx)];
end

function B = local_bump(nelx, nely, cx, cy, w)
[XX, YY] = meshgrid(1:nelx, 1:nely);
B = exp(-(((XX-cx).^2 + (YY-cy).^2)/(2*w^2)));
B(B < 1e-6) = 0;
end

function [v, nz] = local_admissible(v, rho, rhomin, dmax)
%LOCAL_ADMISSIBLE  Zero the direction where rho +/- dmax*|v| would leave the box.
nz = 0;
for it = 1:5
    v = v/max(norm(v), eps);
    bad = (rho - dmax*abs(v) < rhomin) | (rho + dmax*abs(v) > 1);
    if ~any(bad), break, end
    nz = nz + nnz(bad & v ~= 0);
    v(bad) = 0;
end
v = v/max(norm(v), eps);
assert(all(rho - dmax*abs(v) >= rhomin - 1e-15) && all(rho + dmax*abs(v) <= 1 + 1e-15), ...
    'fi_directions:Inadmissible','direction not admissible after masking');
end
