% PROBE_SUPPORT_PLACEMENT
%
% Isolates SUPPORT PLACEMENT from MESH RESOLUTION for the simply-supported (SS)
% benchmark on the 40x5 mesh.
%
% At nely=5 no node lies on y=H/2, so the physical benchmark cannot be
% represented.  build_supports_exact() picks node row round(nely/2)+1 = 4,
% i.e. y = 0.6H -- the pin sits 10% of the beam height ABOVE mid-height.  The
% mirror-image choice is row 3, y = 0.4H.
%
% This probe runs the SAME mesh, the SAME optimizer and the SAME settings with
% the pin placed at row 3 and at row 4, injected through the existing
% cfg.fixed_dofs interface (topopt_freq_exact.m:177-181).  No solver file is
% modified.
%
% If the two designs are mirror images of one another, the asymmetry seen in
% the 40x5 SS benchmark is caused by the off-centre pin, not by the element
% count -- i.e. the discrete 40x5 problem is a DIFFERENT problem, not a coarse
% approximation of the published one.

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));

out_dir = fullfile(this_dir, 'results', 'probe_support_placement');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

nelx = 40; nely = 5; L = 8; H = 1;
nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);

base = struct('support_type','SS','nelx',nelx,'nely',nely,'volfrac',0.5, ...
    'mass_mode','du2007_c1','sensitivity_filter',true,'rmin_elem',2.5, ...
    'n_target',1,'n_modes',4,'mult_tol',1e-3,'outer_max_iter',80, ...
    'outer_tol',1e-6,'inner_max_iter',30,'inner_tol',1e-4, ...
    'move_lim',0.2,'outer_move',0.2,'alpha',0.5,'acceptance_check',false, ...
    'verbose',false);

rows = [3 4];   % y = 0.4H (mirror)  and  y = 0.6H (what the benchmark uses)
R = cell(numel(rows), 1);

fprintf('\n================================================================\n');
fprintf(' SUPPORT-PLACEMENT PROBE -- SS, 40x5, pin row varied only\n');
fprintf('================================================================\n');

for i = 1:numel(rows)
    r = rows(i);
    lm = nodeNrs(r, 1);  rm = nodeNrs(r, end);
    cfg = base;
    cfg.fixed_dofs = unique([2*lm-1; 2*lm; 2*rm-1; 2*rm]);

    rng(0, 'twister');
    [rho, hist] = topopt_freq_exact(cfg);
    R{i} = reshape(rho, nely, nelx);

    y = (r - 1) * (H / nely);
    fprintf(' pin node row %d -> y = %.3f (%.2f H)   final omega_1 = %.4f  vol = %.4f\n', ...
        r, y, y / H, hist.final_omega(1), hist.final_volume);
    writematrix(R{i}, fullfile(out_dir, sprintf('rho_row%d.csv', r)));
end

% Mirror comparison: design(row 3) vs flipud(design(row 4)).
A = R{1};  B = R{2};  Bf = flipud(B);
num = sum((A(:) - mean(A(:))) .* (Bf(:) - mean(Bf(:))));
den = sqrt(sum((A(:) - mean(A(:))).^2) * sum((Bf(:) - mean(Bf(:))).^2));
r_mirror = num / den;

selfA = corr_metric(A, flipud(A));
selfB = corr_metric(B, flipud(B));

fprintf('\n Mid-height self-symmetry of each design (corr(rho, flipud(rho))):\n');
fprintf('   pin at 0.4H : %.4f\n', selfA);
fprintf('   pin at 0.6H : %.4f\n', selfB);
fprintf(' Cross-mirror correlation corr(rho_{0.4H}, flipud(rho_{0.6H})) = %.4f\n', r_mirror);
fprintf(' max |rho_{0.4H} - flipud(rho_{0.6H})| = %.3e\n', max(abs(A(:) - Bf(:))));

fid = fopen(fullfile(out_dir, 'probe_summary.csv'), 'w');
fprintf(fid, 'metric,value\n');
fprintf(fid, 'self_symmetry_pin_0.4H,%.10f\n', selfA);
fprintf(fid, 'self_symmetry_pin_0.6H,%.10f\n', selfB);
fprintf(fid, 'cross_mirror_correlation,%.10f\n', r_mirror);
fprintf(fid, 'max_abs_mirror_difference,%.10e\n', max(abs(A(:) - Bf(:))));
fclose(fid);
fprintf('\n Wrote %s\n\n', out_dir);

function c = corr_metric(X, Y)
    x = X(:) - mean(X(:));  y = Y(:) - mean(Y(:));
    c = sum(x .* y) / sqrt(sum(x.^2) * sum(y.^2));
end
