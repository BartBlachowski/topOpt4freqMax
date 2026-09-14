function m = bg_metrics(rho, nelx, nely, a, b, rminEl)
%BG_METRICS  Bimodality-gap metrics of one element-wise density field.
%
%   rho is NE x 1 in the solver's column-major order e = (i-1)*nely + j
%   (i = column along the length, j = row from the TOP).  a x b is the physical
%   domain, rminEl the effective filter radius in elements.
%
%   Canonical metric (repository): M_nd = mean(4 rho (1-rho))  [Sigmund 2007],
%   recorded by olhoffSolve as aux.Mnd and by olhoffcurrent_run as final_grayness.
rho = rho(:);
NE  = numel(rho);
h   = b/nely;  dx = a/nelx;
R   = reshape(rho, nely, nelx);

m = struct();
m.NE = NE; m.h = h; m.rminEl = rminEl;
m.Mnd          = mean(4*rho.*(1-rho));
m.gray_01_09   = mean(rho > 0.1 & rho < 0.9);
m.gray_02_08   = mean(rho > 0.2 & rho < 0.8);
m.mid_04_06    = mean(rho >= 0.4 & rho <= 0.6);
m.void_le_01   = mean(rho <= 0.1);
m.solid_ge_09  = mean(rho >= 0.9);
m.at_rhomin    = mean(rho <= 1e-3 + 1e-12);
m.at_one       = mean(rho >= 1 - 1e-12);
m.low_unpinned = mean(rho > 1e-3 + 1e-12 & rho <= 0.1);
m.high_unpinned= mean(rho >= 0.9 & rho < 1 - 1e-12);
m.n_gray_01_09 = sum(rho > 0.1 & rho < 0.9);
m.area_gray_01_09 = m.gray_01_09 * a * b;      % physical area of intermediate material
m.area_mid_04_06  = m.mid_04_06 * a * b;
m.hist20_edges = 0:0.05:1;
m.hist20       = histcounts(rho, m.hist20_edges) / NE;
m.hist10       = histcounts(rho, 0:0.1:1) / NE;
% histogram-valley measure: mass in the central 60 % of the range vs the two modes
m.valley_02_08_over_modes = m.gray_02_08 / max(1 - m.gray_02_08, eps);

% ---- interface length of the 0.5 iso-contour of the centroid field ------
xc = ((1:nelx) - 0.5) * dx;  yc = ((1:nely) - 0.5) * h;
C  = contourc(xc, yc, R, [0.5 0.5]);
L = 0; k = 1;
while k < size(C, 2)
    n = C(2, k); seg = C(:, k+1:k+n);
    L = L + sum(sqrt(sum(diff(seg, 1, 2).^2, 1)));
    k = k + n + 1;
end
m.L_iso05 = L;
S  = R >= 0.5;
nv = sum(sum(S(1:end-1, :) ~= S(2:end, :)));   % horizontal edges (length dx)
nh = sum(sum(S(:, 1:end-1) ~= S(:, 2:end)));   % vertical edges (length h)
m.L_edge = nv*dx + nh*h;
m.w_gray_phys = m.area_gray_01_09 / max(m.L_iso05, eps);   % mean gray-band width
m.w_gray_el   = m.w_gray_phys / h;
m.w_mid_phys  = m.area_mid_04_06 / max(m.L_iso05, eps);

% ---- where the intermediate elements sit ---------------------------------
G   = R > 0.1 & R < 0.9;  Sol = R >= 0.9;  Voi = R <= 0.1;
cap = 12;
dS = bg_chebdist(Sol, cap);  dV = bg_chebdist(Voi, cap);
g  = G(:);
if any(g)
    m.gray_adj_solid    = mean(dS(g) <= 1);
    m.gray_adj_void     = mean(dV(g) <= 1);
    m.gray_adj_both     = mean(dS(g) <= 1 & dV(g) <= 1);
    rc = max(1, ceil(rminEl));
    m.gray_within_R_both= mean(dS(g) <= rc & dV(g) <= rc);
    m.gray_bulk_gt2     = mean(dS(g) > 2 & dV(g) > 2);   % no pure phase within 2 elements
    m.gray_bulk_gtR     = mean(dS(g) > rc | dV(g) > rc);  % not inside one filter radius of BOTH phases
    m.gray_mean_dsolid_el = mean(dS(g));  m.gray_mean_dvoid_el = mean(dV(g));
    m.gray_mean_dsolid_phys = m.gray_mean_dsolid_el * h;
    m.gray_mean_dvoid_phys  = m.gray_mean_dvoid_el * h;
    % split of M_nd into the interface band (within one filter radius of both
    % phases), the bulk gray (everything else intermediate) and the near-mode tails
    q = 4*R.*(1-R);
    band = G & (dS <= rc) & (dV <= rc);
    m.Mnd_band  = sum(q(band)) / NE;
    m.Mnd_bulk  = sum(q(G & ~band)) / NE;
    m.Mnd_tails = sum(q(~G)) / NE;
else
    [m.gray_adj_solid, m.gray_adj_void, m.gray_adj_both, m.gray_within_R_both, ...
     m.gray_bulk_gt2, m.gray_bulk_gtR, m.gray_mean_dsolid_el, m.gray_mean_dvoid_el, ...
     m.gray_mean_dsolid_phys, m.gray_mean_dvoid_phys, m.Mnd_band, m.Mnd_bulk] = deal(NaN);
    m.Mnd_tails = m.Mnd;
end
ring = false(nely, nelx); ring([1 end], :) = true; ring(:, [1 end]) = true;
m.gray_on_boundary_ring_share = sum(G(:) & ring(:)) / max(sum(G(:)), 1);
m.boundary_ring_gray_frac     = mean(R(ring) > 0.1 & R(ring) < 0.9);
m.boundary_ring_share_of_NE   = mean(ring(:));
m.gray_by_row  = mean(G, 2)';          % fraction of gray elements in each row (top -> bottom)
m.gray_by_col  = mean(G, 1);           % fraction of gray elements in each column (left -> right)
m.updown_asym  = mean(abs(R - flipud(R)), 'all');
m.leftright_asym = mean(abs(R - fliplr(R)), 'all');
end
