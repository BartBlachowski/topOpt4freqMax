function [N, J_idx, cluster_idx] = detect_multiplicity(lam, n, tol_join, tol_leave, N_prev)
% DETECT_MULTIPLICITY  Multiplicity N of the target eigenvalue, with hysteresis.
%
%   [N, J_idx, cluster_idx] = detect_multiplicity(lam, n, tol_join, tol_leave, N_prev)
%
%   Fig. 1 step 1 of Olhoff & Du (2014): "Detect possible multiplicity N of
%   omega_n".  The paper gives no criterion, so this is a [R] reconstruction --
%   see PLAN_Olhoff2014_exact.md section 1, row R3.
%
%   Clustering is on the EIGENVALUES lam = omega^2 (the quantity the bound
%   constraints (9b) and the sub-eigenvalue problem (12) are written in),
%   relative to lam(n), scanning UPWARD from n:
%
%       join   the cluster when  (lam_j - lam_n)/lam_n <=  tol_join
%       leave  the cluster only when (lam_j - lam_n)/lam_n >  tol_leave
%
%   with tol_join < tol_leave.  The two-level (Schmitt) test is deliberate:
%
%     * A single tight tolerance never fires.  Natural coalescence in these beam
%       problems is 0.3-1.3 % wide on omega (0.6-2.6 % on lambda), so the
%       historical mult_tol = 1e-3 kept N = 1 for entire runs and the simple-
%       eigenvalue sensitivity was used at near-degeneracy, where it is invalid.
%     * A single loose tolerance makes N chatter between 1 and 2 across outer
%       iterations, which re-poses a structurally different subproblem every
%       step and prevents the bimodal optimum from ever being held.
%
%   Hysteresis rule:
%       N = max( N_join, min(N_prev, N_leave) )
%   i.e. the cluster grows as soon as tol_join is met, and shrinks only once a
%   member has moved beyond tol_leave.
%
%   Inputs
%     lam        nModes x 1  ascending eigenvalues (omega^2)
%     n          scalar      1-based target mode index
%     tol_join   scalar      relative lambda tolerance to enter the cluster
%     tol_leave  scalar      relative lambda tolerance to leave  (>= tol_join)
%     N_prev     scalar      multiplicity accepted at the previous outer
%                            iteration (pass 1 on the first iteration, or [] to
%                            disable hysteresis)
%
%   Outputs
%     N            scalar    multiplicity of lam(n), >= 1
%     J_idx        scalar    index of the first mode above the cluster, = n+N,
%                            or 0 if the cluster reaches the end of lam.  This
%                            is the paper's J = n+N of constraint (19b).
%     cluster_idx  1 x N     [n, ..., n+N-1]
%
%   Reference: Olhoff & Du (2014), Eqs. (11), (19b); Fig. 1 step 1.

lam    = lam(:);
nModes = numel(lam);

if n < 1 || n > nModes
    error('detect_multiplicity:TargetOutOfRange', ...
        'n = %d is outside [1, %d].', n, nModes);
end
if nargin < 4 || isempty(tol_leave), tol_leave = tol_join; end
if tol_leave < tol_join
    error('detect_multiplicity:BadTolerances', ...
        'tol_leave (%g) must be >= tol_join (%g).', tol_leave, tol_join);
end
if nargin < 5 || isempty(N_prev), N_prev = 1; end

lam_n = lam(n);
ref   = max(abs(lam_n), realmin);

N_join  = count_run(lam, n, nModes, ref, lam_n, tol_join);
N_leave = count_run(lam, n, nModes, ref, lam_n, tol_leave);

N = max(N_join, min(N_prev, N_leave));
N = max(1, min(N, nModes - n + 1));

cluster_idx = n : n+N-1;

J_idx = n + N;
if J_idx > nModes
    J_idx = 0;      % cluster extends to the end of the computed spectrum
end
end

% -------------------------------------------------------------------------
function cnt = count_run(lam, n, nModes, ref, lam_n, tol)
% Length of the maximal run of consecutive modes from n satisfying the tolerance.
cnt = 1;
for j = n+1 : nModes
    if (lam(j) - lam_n) / ref <= tol
        cnt = cnt + 1;
    else
        break
    end
end
end
