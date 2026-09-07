function [R, Jm_idx, cluster_idx] = detect_multiplicity_below(lam, n, tol_join, tol_leave, R_prev)
% DETECT_MULTIPLICITY_BELOW  Multiplicity R of the eigenvalue just below the target.
%
%   [R, Jm_idx, cluster_idx] = detect_multiplicity_below(lam, n, tol_join, tol_leave, R_prev)
%
%   For the gap problem (Olhoff & Du 2014 Eq. (10)/(20)) the eigenvalue problem
%   may also yield an R-fold eigenvalue
%
%       lam_hat = lam_j,   j = n-R, ..., n-1
%
%   corresponding to the R largest eigenfrequencies constrained from above by
%   beta_1 (footnote *1, p. 281).  This routine clusters DOWNWARD from n-1 with
%   the same two-level hysteresis as DETECT_MULTIPLICITY, referenced to
%   lam(n-1) -- the largest member of the lower cluster, which is the one
%   beta_1 must dominate.
%
%   Inputs
%     lam        nModes x 1  ascending eigenvalues (omega^2)
%     n          scalar      1-based target mode index of the gap problem (n >= 2)
%     tol_join   scalar      relative lambda tolerance to enter the cluster
%     tol_leave  scalar      relative lambda tolerance to leave (>= tol_join)
%     R_prev     scalar      R accepted at the previous outer iteration
%
%   Outputs
%     R            scalar    multiplicity of lam(n-1), >= 1
%     Jm_idx       scalar    index n-R-1 of the first mode BELOW the cluster,
%                            or 0 if the cluster reaches mode 1.  This is the
%                            j = n-R-1 guard of constraint (20e), active only
%                            when R <= n-2.
%     cluster_idx  1 x R     [n-R, ..., n-1]
%
%   Reference: Olhoff & Du (2014), Eq. (10c), (20d)-(20g), footnote *1 p. 281.

lam = lam(:);

if n < 2
    error('detect_multiplicity_below:TargetTooLow', ...
        'The gap problem needs n >= 2 (got n = %d).', n);
end
if nargin < 4 || isempty(tol_leave), tol_leave = tol_join; end
if tol_leave < tol_join
    error('detect_multiplicity_below:BadTolerances', ...
        'tol_leave (%g) must be >= tol_join (%g).', tol_leave, tol_join);
end
if nargin < 5 || isempty(R_prev), R_prev = 1; end

top   = n - 1;                 % highest member of the lower cluster
lam_t = lam(top);
ref   = max(abs(lam_t), realmin);

R_join  = count_run_down(lam, top, ref, lam_t, tol_join);
R_leave = count_run_down(lam, top, ref, lam_t, tol_leave);

R = max(R_join, min(R_prev, R_leave));
R = max(1, min(R, top));

cluster_idx = (top - R + 1) : top;

Jm_idx = top - R;              % = n-R-1
if Jm_idx < 1
    Jm_idx = 0;                % cluster reaches the bottom of the spectrum
end
end

% -------------------------------------------------------------------------
function cnt = count_run_down(lam, top, ref, lam_t, tol)
cnt = 1;
for j = top-1 : -1 : 1
    if (lam_t - lam(j)) / ref <= tol
        cnt = cnt + 1;
    else
        break
    end
end
end
