function [cfg, tgt] = olhoff2014_case(name, overrides)
% OLHOFF2014_CASE  Configuration and published targets for the paper's examples.
%
%   [cfg, tgt] = olhoff2014_case(name)
%   [cfg, tgt] = olhoff2014_case(name, overrides)
%
%   Single place where every number taken from Olhoff & Du (2014) enters the
%   code, so the runners contain no magic constants.
%
%   Cases
%   -----
%     'ss_n1'  'cs_n1'  'cc_n1'    section 3.1 / Figs. 2, 3, 4, 5
%                                  max omega_1, all three BCs
%     'ss_n2'  'cs_n2'  'cc_n2'    section 3.2 / Fig. 6
%                                  max omega_2, all three BCs
%     'cc_gap23'                   section 3.3 / Fig. 7
%                                  max (omega_3 - omega_2), CC beam with a
%                                  concentrated mass m_c = m_b/2 at the
%                                  mid-point of the lower edge
%
%   Shared data, Olhoff & Du (2014) section 3.1 [E14]
%       a = 8, b = 1, plane stress, E = 1e7, nu = 0.3, rho_m = 1,
%       alpha = 50 %, uniform initial density rho = 0.5
%
%   Reconstructed / imported settings are documented in
%   PLAN_Olhoff2014_exact.md section 1 and repeated in cfg for the record.
%
%   `overrides` is an optional struct whose fields replace cfg fields.

if nargin < 2, overrides = struct(); end

% ---- Shared [E14] -------------------------------------------------------
cfg = struct();
cfg.L = 8;  cfg.H = 1;  cfg.t = 1;
cfg.E0 = 1e7;  cfg.nu = 0.3;  cfg.rho0 = 1;
cfg.volfrac = 0.5;
cfg.rho_min = 1e-3;                      % [E3] section 2.1
% ---- Imported from Du & Olhoff (2007) [D1-D5] ---------------------------
cfg.penal              = 3;              % [D1]
cfg.mass_mode          = 'du2007_c1';    % [D3] Eq. (4b)
cfg.mass_q             = 1;              % [D2]
cfg.sensitivity_filter = true;           % [D4]
% Filter RADIUS is [R], not [D]: Olhoff2014 states no filter at all, so nothing
% fixes it.  Calibrated on the SS case only (experiments/ablations, 160x20, 800
% iterations); CS and CC are then predictions at this fixed value.
%   rmin  1.0   1.2    1.5    2.0    2.5
%   w1  114.0  168.7  166.1  161.8  158.9      (paper 174.7)
% rmin = 1.0 gives a 1x1 kernel, i.e. no filtering, and collapses -- identical
% to the sensitivity_filter = false arm to the last digit.
cfg.rmin_elem          = 1.2;            % [R] calibrated, was 2.5 from Du2007
% ---- Reconstructed [R1-R9] ---------------------------------------------
cfg.nelx = 160;  cfg.nely = 20;          % [R2] mesh unspecified in the paper
cfg.move               = 0.05;           % [R1] calibrated, see experiments/step_calibration
cfg.mult_tol_join      = 2e-2;           % [R3] on lambda
cfg.mult_tol_leave     = 5e-2;           % [R3]
cfg.cluster_model      = 'CA';           % [R4]
cfg.lam_ref_rule       = 'lowest';       % [R4]
cfg.subproblem_solver  = 'lp';           % [R9] exact; 'mma' is the E13 alternative
cfg.outer_max_iter     = 800;
cfg.outer_tol          = 1e-4;           % [R8]
cfg.kkt_tol            = 1e-6;           % [R8]
cfg.N_max              = 3;
cfg.verbose            = true;
cfg.rho_snapshot_interval = 10;

tgt = struct();
tgt.name = name;

switch lower(strtrim(name))

    % ================================================= section 3.1, Fig. 3
    case 'ss_n1'
        cfg.support_type = 'SS';  cfg.objective = 'nth';  cfg.n_target = 1;
        tgt.omega_initial = 68.7;      % Fig. 2 caption
        tgt.omega_opt     = 174.7;     % Fig. 3a
        tgt.multiplicity  = 2;         % "all found to be bimodal"
        tgt.extra         = struct('omega3_opt', 284.9);   % Fig. 5c
        tgt.increase_pct  = 154;       % Fig. 3 caption
        tgt.figure        = 'Fig. 3a / 4 / 5';

    case 'cs_n1'
        cfg.support_type = 'CS';  cfg.objective = 'nth';  cfg.n_target = 1;
        tgt.omega_initial = 104.1;
        tgt.omega_opt     = 288.7;
        tgt.multiplicity  = 2;
        tgt.increase_pct  = 177;
        tgt.figure        = 'Fig. 3b';

    case 'cc_n1'
        cfg.support_type = 'CC';  cfg.objective = 'nth';  cfg.n_target = 1;
        tgt.omega_initial = 146.1;
        tgt.omega_opt     = 456.4;
        tgt.multiplicity  = 2;
        tgt.increase_pct  = 212;
        tgt.figure        = 'Fig. 3c';

    % ================================================= section 3.2, Fig. 6
    case 'ss_n2'
        cfg.support_type = 'SS';  cfg.objective = 'nth';  cfg.n_target = 2;
        tgt.omega_opt    = 598.3;
        tgt.multiplicity = 2;
        tgt.figure       = 'Fig. 6a';

    case 'cs_n2'
        cfg.support_type = 'CS';  cfg.objective = 'nth';  cfg.n_target = 2;
        tgt.omega_opt    = 732.8;
        tgt.multiplicity = 2;
        tgt.figure       = 'Fig. 6b';

    case 'cc_n2'
        cfg.support_type = 'CC';  cfg.objective = 'nth';  cfg.n_target = 2;
        tgt.omega_opt    = 849.0;
        tgt.multiplicity = 2;
        tgt.figure       = 'Fig. 6c';

    % ================================================= section 3.3, Fig. 7
    case 'cc_gap23'
        cfg.support_type      = 'CC';
        cfg.objective         = 'gap';
        cfg.n_target          = 3;        % maximize omega_3 - omega_2
        cfg.lumped_mass_frac  = 0.5;      % m_c = m_b/2, m_b = initial total mass
        cfg.lumped_mass       = struct('where','bottom_mid');
        cfg.n_modes           = 8;
        cfg.outer_max_iter    = 200;      % Fig. 7c shows 100 iterations
        tgt.gap_opt           = 810;      % section 3.3
        tgt.increase_pct      = 548;
        tgt.multiplicity      = 3;        % omega_3, omega_4, omega_5 trimodal
        tgt.figure            = 'Fig. 7';

    otherwise
        error('olhoff2014_case:UnknownCase', ...
            ['Unknown case ''%s''.  Use ss_n1 | cs_n1 | cc_n1 | ss_n2 | cs_n2 | ' ...
             'cc_n2 | cc_gap23.'], name);
end

% n_modes default depends on n_target; set after the switch unless overridden.
if ~isfield(cfg, 'n_modes')
    cfg.n_modes = max(cfg.n_target + cfg.N_max + 3, 6);
end

% ---- Apply caller overrides -------------------------------------------
fn = fieldnames(overrides);
for k = 1:numel(fn)
    cfg.(fn{k}) = overrides.(fn{k});
end
end
