function c = a4_phase2_constants()
%A4_PHASE2_CONSTANTS  Normative constants for A4 Recovery Phase 2.
% Source of truth: A4_RECOVERY_PHASE2_SPECIFICATION.md §§3.2, 3.6, 4.1.
% Keep every protocol constant here; changing any value requires a written
% specification amendment and a complete five-arm rerun (spec §11 R-4).

c = struct();
c.window_ladder = [20, 40, 80, 160, 320];
c.m0 = 20;
c.M_max = 320;
c.tau_mac = 0.80;
c.tau_stab = 0.99;
c.tau_kin = 0.50;
c.tau_strain = 0.50;
c.x_low = 0.10;
c.tie_tolerance = 1e-12;
c.solid_threshold = 0.50;
c.diagnostic_grid = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, ...
    75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000];

% Fixed eigensolver settings (§2.2). A numeric sigma of zero requests the
% lowest modes through MATLAB's generalized shift-invert eigensolver.
c.eigensolver = struct('sigma', 0, 'tolerance', 1e-12, ...
    'max_iterations', 2000, 'display', 0);
end
