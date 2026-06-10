function [N, J_idx, cluster_idx] = detect_multiplicity(omega, n, mult_tol)
% DETECT_MULTIPLICITY  Find the multiplicity of eigenfrequency omega(n).
%
%   [N, J_idx, cluster_idx] = detect_multiplicity(omega, n, mult_tol)
%
%   Scans upward from index n in the sorted vector omega and groups all
%   consecutive eigenfrequencies satisfying
%
%       |omega(j) - omega(n)| / max(omega(n), eps) <= mult_tol
%
%   into a single cluster.  This implements the multiplicity detection step
%   described in Du & Olhoff (2007), Section 3.5.1.
%
%   Inputs:
%     omega     nModes x 1   sorted eigenfrequencies (rad/s, from eigs)
%     n         scalar       1-based index of the target mode
%     mult_tol  scalar       relative frequency tolerance, e.g. 1e-3
%
%   Outputs:
%     N           scalar      multiplicity of omega(n) (>= 1)
%     J_idx       scalar      index of the first simple mode above the cluster
%                             (= n+N if available in omega, else 0)
%     cluster_idx 1 x N       indices [n, n+1, ..., n+N-1] of the cluster
%
%   Notes:
%     - omega must be sorted in ascending order before calling.
%     - For N = 1 (simple eigenvalue), cluster_idx = [n] and J_idx = n+1.
%     - The mode at J_idx is not verified to be simple; it is assumed to be
%       so when its frequency differs from omega(n) by more than mult_tol.
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110.

omega = omega(:);
nModes = numel(omega);

if n < 1 || n > nModes
    error('detect_multiplicity: n=%d is out of range [1, %d].', n, nModes);
end

omega_n = omega(n);
ref     = max(omega_n, eps);   % avoid division by zero for near-zero frequencies

% Count consecutive modes in the cluster starting at n.
N = 1;
for j = n+1 : nModes
    if abs(omega(j) - omega_n) / ref <= mult_tol
        N = N + 1;
    else
        break
    end
end

cluster_idx = n : n+N-1;   % 1 x N row vector

% Index of first mode above the cluster.
J_idx = n + N;
if J_idx > nModes
    J_idx = 0;   % cluster extends to the end of the computed set
end
end
