function out = fi_filter()
%FI_FILTER  Part 8: analytic structure of the sensitivity-filter operator.
%
%   The production filter (applyFilter.m) is, elementwise,
%       g_filt(e) = sum_i H_ei * rho_i * g_phys(i) / ( Hs_e * max(1e-3, rho_e) )
%   On the admissible set rho >= rhomin = 1e-3, so max(1e-3,rho) = rho and
%       g_filt = A(rho) * g_phys,      A_ei = H_ei * rho_i / (Hs_e * rho_e).
%   Everything below is exact linear algebra on that A; no finite differences.

S = fi_setup();
study = fileparts(fileparts(mfilename('fullpath')));
rho = S.rho386(:);
H = S.flt.H; Hs = full(S.flt.Hs(:));

out = struct();
out.formula = 'g_filt = diag(1./(Hs.*rho)) * H * diag(rho) * g_phys';
out.rhomin = S.rhomin;
out.max_guard_active = sum(rho < 1e-3);      % where max(1e-3,rho) differs from rho
out.H_symmetric = issymmetric(H);
out.H_nnz = nnz(H);
out.H_rowsum_equals_Hs = full(max(abs(sum(H,2) - Hs)));

% ---- A and its asymmetry, exactly --------------------------------------
A = spdiags(1./(Hs.*rho),0,S.NE,S.NE) * H * spdiags(rho,0,S.NE,S.NE);
Askew = A - A.';
out.A = struct('nnz',nnz(A), ...
    'symmetric', isequal(A, A.'), ...
    'skew_fro', full(norm(Askew,'fro')), ...
    'A_fro', full(norm(A,'fro')), ...
    'skew_ratio', full(norm(Askew,'fro'))/full(norm(A,'fro')), ...
    'rowsum_min', full(min(sum(A,2))), 'rowsum_max', full(max(sum(A,2))));

% A row sums equal rho_tilde/rho with rho_tilde = (H*rho)/Hs
rhoT = (H*rho)./Hs;
out.A.rowsum_identity_maxerr = full(max(abs(sum(A,2) - rhoT./rho)));
out.rho_tilde = struct('min',min(rhoT),'max',max(rhoT),'mean',mean(rhoT));

% ---- exact symmetry condition: rho_i^2*Hs_i == rho_e^2*Hs_e on the stencil
w = rho.^2 .* Hs;
[ii,jj] = find(H);
sel = ii ~= jj;
ii = ii(sel); jj = jj(sel);
rel = abs(w(ii)-w(jj))./max(abs(w(ii)),abs(w(jj)));
out.symmetry_condition = struct( ...
    'statement','A is symmetric iff rho_i^2*Hs_i = rho_e^2*Hs_e for every pair with H_ei ~= 0', ...
    'n_offdiag_pairs', numel(ii), ...
    'median_relative_violation', median(rel), ...
    'p90_relative_violation', quantile(rel,0.90), ...
    'max_relative_violation', max(rel), ...
    'frac_below_1e_6', mean(rel < 1e-6));

% ---- analytic skew of term 1 (needs NO finite differences) ---------------
% J = A*D_{g/rho} + A*Hess - diag(g_filt/rho);  the last term is diagonal.
% T1_ej = H_ej * g_j / (Hs_e * rho_e)   ->  skew1 = T1 - T1'
E = fi_eval(S, rho, 'mode1');
g = E.gPhys;
T1 = spdiags(1./(Hs.*rho),0,S.NE,S.NE) * H * spdiags(g,0,S.NE,S.NE);
T1skew = T1 - T1.';
out.term1 = struct('T1_fro', full(norm(T1,'fro')), ...
    'skew_fro', full(norm(T1skew,'fro')), ...
    'skew_ratio', full(norm(T1skew,'fro'))/full(norm(T1,'fro')), ...
    'max_abs_skew', full(max(abs(T1skew(:)))));

% ---- volume constraint: constant gradient, so no effect on any curl -----
out.volume = struct( ...
    'gradient_value', 1/S.Vtot, ...
    'is_constant_in_rho', true, ...
    'statement', ['the volume row gradient is 1/Vtot on every element and does ' ...
                  'not depend on rho, so it adds a CONSTANT vector field; the ' ...
                  'Jacobian of a constant field is zero and the antisymmetric ' ...
                  'part of J is therefore unchanged'], ...
    'proof', 'd/drho_j [ lam_V / Vtot ] = 0 for all j, hence skew(J_reduced) = skew(J_filt)');

% ---- does H*g_phys alone (unweighted) fare better? ----------------------
% A purely symmetric averaging operator B = diag(1/Hs)*H is NOT symmetric either
% unless Hs is constant; recorded because it is the usual informal claim.
B = spdiags(1./Hs,0,S.NE,S.NE)*H;
out.plainAverage = struct('symmetric', isequal(B,B.'), ...
    'skew_ratio', full(norm(B-B.','fro'))/full(norm(B,'fro')), ...
    'Hs_constant', full(max(Hs)-min(Hs)) < 1e-12, ...
    'Hs_min', full(min(Hs)), 'Hs_max', full(max(Hs)));

f = fullfile(study,'evaluations','filter_operator.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);

fprintf('[fi_filter]\n');
fprintf('  H symmetric = %d, nnz = %d, rowsum==Hs err = %.3g\n', out.H_symmetric, out.H_nnz, out.H_rowsum_equals_Hs);
fprintf('  max(1e-3,rho) guard active on %d elements\n', out.max_guard_active);
fprintf('  A symmetric = %d ; ||A-A''||_F/||A||_F = %.4f\n', out.A.symmetric, out.A.skew_ratio);
fprintf('  A row sums = rho_tilde/rho  (identity err %.3g), range [%.4f, %.4f]\n', ...
    out.A.rowsum_identity_maxerr, out.A.rowsum_min, out.A.rowsum_max);
fprintf('  symmetry condition rho^2*Hs equal on stencil: median viol %.4f, p90 %.4f, max %.4f, frac<1e-6 = %.4g\n', ...
    out.symmetry_condition.median_relative_violation, out.symmetry_condition.p90_relative_violation, ...
    out.symmetry_condition.max_relative_violation, out.symmetry_condition.frac_below_1e_6);
fprintf('  term1 skew ratio = %.4f (analytic, no FD)\n', out.term1.skew_ratio);
fprintf('  plain averaging diag(1/Hs)*H symmetric = %d (skew ratio %.4f), Hs range [%.3f, %.3f]\n', ...
    out.plainAverage.symmetric, out.plainAverage.skew_ratio, out.plainAverage.Hs_min, out.plainAverage.Hs_max);
end
