% RUN_MISSING_REGULARIZATION_DIAGNOSTICS
%
% Diagnostic-only search for implicit regularization or benchmark assumptions
% missing from the OlhoffApproachExact CC benchmark reconstruction.
%
% The default paper-like optimizer path is left unchanged.  Variants below
% intentionally add or swap assumptions such as density filtering, Heaviside
% projection, lumped mass, support-placement alternatives, and a passive
% support path.  Any successful variant using those additions is classified as
% paper-ambiguous or Olhoff-inspired, not paper-faithful reproduction.

clearvars;
close all;
clc;

this_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(this_dir, '..', '..', '..');
matlab_dir = fullfile(this_dir, '..', 'Matlab');
tools_dir = fullfile(repo_root, 'tools', 'Matlab');
addpath(matlab_dir);
addpath(tools_dir);

out_dir = fullfile(this_dir, 'missing_regularization_diagnostics_results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

paper_omega = 456.4;
paper_tol = 0.02;
gap_tol = 0.005;
support_energy_tol = 0.50;

variants = build_variants();
summaries = cell(numel(variants), 1);

fprintf('\n==========================================================\n');
fprintf(' Missing-regularization diagnostic variants\n');
fprintf(' Output: %s\n', out_dir);
fprintf('==========================================================\n\n');

for iv = 1:numel(variants)
    v = variants(iv);
    variant_dir = fullfile(out_dir, v.tag);
    if ~exist(variant_dir, 'dir'), mkdir(variant_dir); end
    audit_path = fullfile(variant_dir, [v.tag '_audit.mat']);
    if exist(audit_path, 'file')
        loaded = load(audit_path, 'summary');
        summaries{iv} = loaded.summary;
        fprintf('Reusing %-36s  %s\n', v.tag, v.description);
        continue
    end
    fprintf('Running %-36s  %s\n', v.tag, v.description);
    t0 = tic;
    [rho_final, hist, stage_info, final_cfg] = run_variant(v);
    elapsed_s = toc(t0);
    audit = audit_final_design(rho_final, final_cfg);
    summary = summarize(v, hist, rho_final, audit, stage_info, elapsed_s, ...
        paper_omega, paper_tol, gap_tol, support_energy_tol, variant_dir);
    summaries{iv} = summary;

    write_component_table(audit, fullfile(variant_dir, [v.tag '_components.csv']));
    write_mode_energy_table(audit, fullfile(variant_dir, [v.tag '_mode_component_energy.csv']));
    writematrix(rho_final(:), fullfile(variant_dir, [v.tag '_rho_final.csv']));
    write_iteration_table(hist, fullfile(variant_dir, [v.tag '_iterations.csv']));
    export_topology(rho_final, final_cfg, summary, fullfile(variant_dir, [v.tag '_topology.png']));
    save(fullfile(variant_dir, [v.tag '_audit.mat']), ...
        'v', 'final_cfg', 'hist', 'stage_info', 'rho_final', 'audit', 'summary', 'elapsed_s');

    fprintf(['  omega=(%.3f, %.3f) gap=%.3g N=%g vol=%.4f comps=%d ', ...
        'support=%d supportE=(%.2f, %.2f) success=%d class=%s\n'], ...
        summary.omega1, summary.omega2, summary.gap12, summary.N, summary.volume, ...
        summary.component_count, summary.support_connected, summary.mode1_support_energy, ...
        summary.mode2_support_energy, summary.success, summary.classification);
end

summary_struct = [summaries{:}];
summary_table = summaries_to_table(summary_struct);
writetable(summary_table, fullfile(out_dir, 'missing_regularization_diagnostics_summary.csv'));
save(fullfile(out_dir, 'missing_regularization_diagnostics_results.mat'), ...
    'summary_struct', 'summary_table', 'paper_omega', 'paper_tol', 'gap_tol', ...
    'support_energy_tol');
write_report(fullfile(this_dir, 'missing_regularization_diagnostics_report.md'), ...
    summary_struct, paper_omega, paper_tol, gap_tol, support_energy_tol);

fprintf('\nReport written to %s\n', fullfile(this_dir, 'missing_regularization_diagnostics_report.md'));

function variants = build_variants()
items = {};
items{end+1} = make_variant('baseline_sensitivity_full_edges', ...
    'baseline stabilized sensitivity filter, full-edge CC', @(c) c, 'paper-faithful');
items{end+1} = make_variant('density_filter_r2p5', ...
    'density filter rmin=2.5', @(c) cfg_filter(c, 'density', 2.5), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('density_filter_r5p0', ...
    'density filter rmin=5.0', @(c) cfg_filter(c, 'density', 5.0), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('density_filter_r7p5', ...
    'density filter rmin=7.5', @(c) cfg_filter(c, 'density', 7.5), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('projection_cont_r2p5_sym_both', ...
    'density filter + Heaviside continuation rmin=2.5 + both symmetry', @(c) cfg_projection(c, 2.5, 'both'), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('projection_cont_r5p0_sym_both', ...
    'density filter + Heaviside continuation rmin=5.0 + both symmetry', @(c) cfg_projection(c, 5.0, 'both'), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('projection_cont_r7p5_sym_both', ...
    'density filter + Heaviside continuation rmin=7.5 + both symmetry', @(c) cfg_projection(c, 7.5, 'both'), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('support_path_sensitivity_r2p5', ...
    'minimum passive solid support path + sensitivity filter', @(c) cfg_support_path(c, 'sensitivity', 2.5), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('support_path055_sensitivity_r2p5', ...
    'minimum threshold support path rho=0.55 + sensitivity filter', @(c) cfg_support_path_value(c, 'sensitivity', 2.5, 0.55), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('support_path055_sensitivity_r3p5', ...
    'minimum threshold support path rho=0.55 + sensitivity filter rmin=3.5', @(c) cfg_support_path_value(c, 'sensitivity', 3.5, 0.55), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('support_path055_sensitivity_alpha1', ...
    'threshold support path rho=0.55 + alpha=1, trust off', @(c) cfg_support_path_aggressive(c, 0.55, 1.0, 0.2), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('support_path055_sensitivity_move0p5', ...
    'threshold support path rho=0.55 + move=0.5, alpha=0.5, trust off', @(c) cfg_support_path_aggressive(c, 0.55, 0.5, 0.5), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('support_path055_sensitivity_alpha1_move0p5', ...
    'threshold support path rho=0.55 + move=0.5, alpha=1, trust off', @(c) cfg_support_path_aggressive(c, 0.55, 1.0, 0.5), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('support_path055_density_r7p5', ...
    'minimum threshold support path rho=0.55 + density filter rmin=7.5', @(c) cfg_support_path_value(c, 'density', 7.5, 0.55), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('support_path_density_r5p0', ...
    'minimum passive solid support path + density filter rmin=5.0', @(c) cfg_support_path(c, 'density', 5.0), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('support_path_projection_cont_r5p0', ...
    'minimum passive solid support path + projection continuation rmin=5.0', @(c) cfg_support_path_projection(c, 5.0), 'Olhoff-inspired stabilized reconstruction');
items{end+1} = make_variant('mass_lumped_sensitivity', ...
    'row-sum lumped Q4 mass + sensitivity filter', @cfg_lumped, 'paper-ambiguous');
items{end+1} = make_variant('mass_lumped_density_r5p0', ...
    'row-sum lumped Q4 mass + density filter rmin=5.0', @(c) cfg_lumped_density(c, 5.0), 'paper-ambiguous');
items{end+1} = make_variant('support_corner_clamps', ...
    'paper-figure alternative: only end-corner translational clamps', @(c) cfg_support_alt(c, 'corner_clamps'), 'paper-ambiguous');
items{end+1} = make_variant('support_midheight_clamps', ...
    'paper-figure alternative: only midheight end translational clamps', @(c) cfg_support_alt(c, 'midheight_clamps'), 'paper-ambiguous');
variants = [items{:}];
end

function v = make_variant(tag, description, modifier, success_class)
cfg = cfg_baseline();
cfg = modifier(cfg);
v = struct('tag', tag, 'description', description, 'cfg', cfg, ...
    'success_class', success_class, 'continuation', false, ...
    'projection_continuation', false);
if isfield(cfg, 'run_projection_continuation') && cfg.run_projection_continuation
    v.continuation = true;
    v.projection_continuation = true;
end
end

function cfg = cfg_baseline()
cfg = struct();
cfg.support_type = 'CC';
cfg.L = 8.0; cfg.H = 1.0;
cfg.nelx = 40; cfg.nely = 5;
cfg.E0 = 1e7; cfg.nu = 0.3; cfg.rho0 = 1.0; cfg.t = 1.0;
cfg.volfrac = 0.5; cfg.rho_min = 1e-3;
cfg.mass_mode = 'du2007_c1';
cfg.mass_matrix = 'consistent';
cfg.filter_type = 'sensitivity';
cfg.sensitivity_filter = true;
cfg.rmin_elem = 2.5;
cfg.n_target = 1; cfg.n_modes = 6; cfg.mult_tol = 1e-3;
cfg.penal = 3;
cfg.outer_max_iter = 100; cfg.outer_tol = 1e-6;
cfg.inner_max_iter = 30; cfg.inner_tol = 1e-4;
cfg.move_lim = 0.2; cfg.outer_move = 0.2; cfg.alpha = 0.5;
cfg.verbose = false; cfg.rho_snapshot_interval = 0;
cfg.post_coalescence_trust_enabled = true;
cfg.post_coalescence_trust_factor = 0.1;
cfg.post_coalescence_gap_tol = 0.005;
cfg.globalization_enabled = false;
cfg.persistent_mma_state = false;
cfg.density_symmetry = 'none';
end

function cfg = cfg_filter(cfg, filter_type, rmin)
cfg.filter_type = filter_type;
cfg.sensitivity_filter = strcmp(filter_type, 'sensitivity');
cfg.rmin_elem = rmin;
end

function cfg = cfg_projection(cfg, rmin, symmetry)
cfg.filter_type = 'density_projection';
cfg.sensitivity_filter = false;
cfg.rmin_elem = rmin;
cfg.projection_eta = 0.5;
cfg.projection_beta = 1;
cfg.density_symmetry = symmetry;
cfg.run_projection_continuation = true;
end

function cfg = cfg_support_path(cfg, filter_type, rmin)
cfg = cfg_filter(cfg, filter_type, rmin);
cfg.forced_solid_mask = support_path_mask(cfg.nelx, cfg.nely);
cfg.forced_solid_preserve_volume = true;
cfg.initial_rho = initial_with_forced_path(cfg);
end

function cfg = cfg_support_path_value(cfg, filter_type, rmin, value)
cfg = cfg_filter(cfg, filter_type, rmin);
cfg.forced_solid_mask = support_path_mask(cfg.nelx, cfg.nely);
cfg.forced_solid_value = value;
cfg.forced_solid_preserve_volume = true;
cfg.initial_rho = initial_with_forced_path(cfg);
end

function cfg = cfg_support_path_aggressive(cfg, value, alpha, move)
cfg = cfg_support_path_value(cfg, 'sensitivity', 2.5, value);
cfg.alpha = alpha;
cfg.move_lim = move;
cfg.outer_move = move;
cfg.post_coalescence_trust_enabled = false;
cfg.outer_max_iter = 120;
end

function cfg = cfg_support_path_projection(cfg, rmin)
cfg = cfg_projection(cfg, rmin, 'both');
cfg.forced_solid_mask = support_path_mask(cfg.nelx, cfg.nely);
cfg.forced_solid_preserve_volume = true;
cfg.initial_rho = initial_with_forced_path(cfg);
end

function cfg = cfg_lumped(cfg)
cfg.mass_matrix = 'lumped';
end

function cfg = cfg_lumped_density(cfg, rmin)
cfg.mass_matrix = 'lumped';
cfg = cfg_filter(cfg, 'density', rmin);
end

function cfg = cfg_support_alt(cfg, alt)
nodeNrs = reshape(1:(cfg.nelx+1)*(cfg.nely+1), cfg.nely+1, cfg.nelx+1);
u = @(n) 2*n - 1;
v = @(n) 2*n;
switch alt
    case 'corner_clamps'
        nodes = [nodeNrs(1,1); nodeNrs(end,1); nodeNrs(1,end); nodeNrs(end,end)];
    case 'midheight_clamps'
        mid = round(cfg.nely/2) + 1;
        nodes = [nodeNrs(mid,1); nodeNrs(mid,end)];
    otherwise
        error('Unknown support alternative: %s', alt);
end
cfg.fixed_dofs = unique([u(nodes(:)); v(nodes(:))]);
cfg.support_alt = alt;
end

function mask = support_path_mask(nelx, nely)
mask = false(nely, nelx);
row = ceil(nely / 2);
mask(row, :) = true;
mask = mask(:);
end

function rho0 = initial_with_forced_path(cfg)
mask = cfg.forced_solid_mask(:);
n = numel(mask);
forced_value = 1.0;
if isfield(cfg, 'forced_solid_value') && ~isempty(cfg.forced_solid_value)
    forced_value = cfg.forced_solid_value;
end
target_free = (cfg.volfrac * n - forced_value * nnz(mask)) / max(nnz(~mask), 1);
target_free = max(cfg.rho_min, min(1, target_free));
rho0 = target_free * ones(n, 1);
rho0(mask) = forced_value;
end

function [rho_final, hist, stage_info, final_cfg] = run_variant(v)
cfg = v.cfg;
stage_info = struct('penal', {}, 'projection_beta', {}, 'outer_iters', {}, ...
    'omega1', {}, 'omega2', {});
if v.projection_continuation
    penals = [1, 2, 3, 3];
    betas = [1, 2, 4, 8];
    iters = [35, 35, 50, 80];
    rho_design = [];
    if isfield(cfg, 'initial_rho') && ~isempty(cfg.initial_rho)
        rho_design = cfg.initial_rho;
    end
    for i = 1:numel(penals)
        cfg.penal = penals(i);
        cfg.projection_beta = betas(i);
        cfg.outer_max_iter = iters(i);
        cfg.initial_rho = rho_design;
        [rho_stage, hist_stage] = topopt_freq_exact(cfg); %#ok<ASGLU>
        if isfield(hist_stage, 'rho_design_final')
            rho_design = hist_stage.rho_design_final;
        else
            rho_design = rho_stage;
        end
        stage_info(i).penal = penals(i);
        stage_info(i).projection_beta = betas(i);
        stage_info(i).outer_iters = hist_stage.outer_iters;
        stage_info(i).omega1 = hist_stage.final_omega(1);
        stage_info(i).omega2 = hist_stage.final_omega(2);
    end
    rho_final = hist_stage.rho_final;
    hist = hist_stage;
    final_cfg = cfg;
else
    [rho_final, hist] = topopt_freq_exact(cfg);
    final_cfg = cfg;
    stage_info(1).penal = cfg.penal;
    stage_info(1).projection_beta = NaN;
    stage_info(1).outer_iters = hist.outer_iters;
    stage_info(1).omega1 = hist.final_omega(1);
    stage_info(1).omega2 = hist.final_omega(2);
end
end

function audit = audit_final_design(rho, cfg)
[omega, lam, Phi, cMat, Ke_phys, Me_phys, flag] = eval_modes_full(rho, cfg);
if flag ~= 0
    error('run_missing_regularization_diagnostics:eigs', 'Final eigensolve failed.');
end
[comp_id, conn] = component_map(rho, cfg);
[mcoef, ~] = mass_interp(rho, cfg.mass_mode);
kcoef = rho(:).^cfg.penal;
num_comp = max(comp_id);
num_modes = min(6, numel(omega));
kin_frac = zeros(num_comp, num_modes);
strain_frac = zeros(num_comp, num_modes);
total_frac = zeros(num_comp, num_modes);
for j = 1:num_modes
    phi = Phi(:, j);
    elemK = zeros(numel(rho), 1);
    elemM = zeros(numel(rho), 1);
    for e = 1:numel(rho)
        dofs = cMat(e,:);
        pe = phi(dofs);
        elemK(e) = kcoef(e) * (pe' * Ke_phys * pe);
        elemM(e) = mcoef(e) * (pe' * Me_phys * pe);
    end
    for c = 1:num_comp
        idx = comp_id == c;
        kin_frac(c,j) = sum(elemM(idx)) / max(sum(elemM), eps);
        strain_frac(c,j) = sum(elemK(idx)) / max(sum(elemK), eps);
        total_frac(c,j) = 0.5 * (kin_frac(c,j) + strain_frac(c,j));
    end
end
support_components = find(conn.component_touches_left & conn.component_touches_right);
support_total = zeros(1, num_modes);
for j = 1:num_modes
    if ~isempty(support_components)
        support_total(j) = sum(total_frac(support_components, j));
    end
end
audit = struct('omega', omega(:)', 'lambda', lam(:)', 'Phi', Phi, ...
    'component_id', comp_id, 'connectedness', conn, ...
    'kinetic_fraction', kin_frac, 'strain_fraction', strain_frac, ...
    'total_fraction', total_frac, 'support_total_fraction', support_total);
end

function s = summarize(v, hist, rho, audit, stage_info, elapsed_s, paper_omega, paper_tol, gap_tol, support_energy_tol, variant_dir)
omega = hist.final_omega(:)';
gap12 = abs(omega(2) - omega(1)) / max(omega(1), eps);
paper_close = abs(omega(1) - paper_omega) / paper_omega <= paper_tol;
coalesced = gap12 <= gap_tol;
support_connected = audit.connectedness.support_connected;
mode_support_ok = audit.support_total_fraction(1) >= support_energy_tol && ...
    audit.support_total_fraction(2) >= support_energy_tol;
success = paper_close && coalesced && support_connected && mode_support_ok && hist.final_N >= 2;
if success
    cls = v.success_class;
else
    cls = 'unsuccessful';
end
s = struct();
s.tag = v.tag;
s.description = v.description;
s.elapsed_s = elapsed_s;
s.omega1 = omega(1);
s.omega2 = omega(2);
s.gap12 = gap12;
s.N = hist.final_N;
s.paper_close = paper_close;
s.coalesced = coalesced;
s.volume = mean(rho);
s.rho_min = min(rho);
s.rho_max = max(rho);
s.component_count = audit.connectedness.component_count;
s.connected = audit.connectedness.connected;
s.support_connected = support_connected;
s.isolated_fraction = audit.connectedness.isolated_solid_fraction;
s.mode1_support_energy = audit.support_total_fraction(1);
s.mode2_support_energy = audit.support_total_fraction(2);
s.mode1_top_component = max(audit.total_fraction(:,1));
s.mode2_top_component = max(audit.total_fraction(:,2));
s.success = success;
s.classification = cls;
s.stage_info = stage_info;
s.variant_dir = variant_dir;
s.topology_png = fullfile(variant_dir, [v.tag '_topology.png']);
end

function [omega, lam, Phi, cMat, Ke_phys, Me_phys, flag] = eval_modes_full(rho, cfg)
nelx = cfg.nelx; nely = cfg.nely;
dx = cfg.L / nelx; dy = cfg.H / nely;
nEl = nelx * nely;
nDof = 2 * (nelx+1) * (nely+1);
[Ke_star, Me_star] = fe_q4_exact(cfg.nu, cfg.t, dx, dy);
Ke_phys = cfg.E0 * Ke_star;
Me_phys = cfg.rho0 * Me_star;
if isfield(cfg, 'mass_matrix') && strcmpi(cfg.mass_matrix, 'lumped')
    Me_phys = diag(sum(Me_phys, 2));
end
nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
        cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
[Il, Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)', [], 1);
jK = reshape(cMat(:,Jl)', [], 1);
Ke_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_l = Me_phys(sub2ind([8,8], Il, Jl));
if isfield(cfg, 'fixed_dofs') && ~isempty(cfg.fixed_dofs)
    fixed = unique(double(cfg.fixed_dofs(:)));
else
    fixed = build_supports_exact(cfg.support_type, nodeNrs);
end
free = setdiff(1:nDof, fixed);
[K, M] = assemble_KM_exact(rho, Ke_l, Me_l, iK, jK, nDof, cfg.penal, cfg.mass_mode);
Kf = K(free, free);
Mf = M(free, free);
opts.tol = 1e-10; opts.maxit = 600;
[V, D, flag] = eigs(Kf, Mf, cfg.n_modes, 'SM', opts);
if flag ~= 0
    opts.tol = 1e-8; opts.maxit = 1500; opts.p = min(numel(free)-1, max(40, 4*cfg.n_modes));
    [V, D, flag] = eigs(Kf, Mf, cfg.n_modes, 'SM', opts);
end
if flag ~= 0
    omega = nan(cfg.n_modes, 1); lam = omega; Phi = nan(nDof, cfg.n_modes); return
end
[lam, idx] = sort(real(diag(D)));
V = real(V(:, idx));
for j = 1:cfg.n_modes
    sc = sqrt(abs(V(:,j)' * (Mf * V(:,j))));
    if sc > 1e-14, V(:,j) = V(:,j) / sc; end
end
omega = sqrt(max(lam, 0));
Phi = zeros(nDof, cfg.n_modes);
for j = 1:cfg.n_modes, Phi(free, j) = V(:, j); end
end

function [comp_id, c] = component_map(rho, cfg)
solid = reshape(rho, cfg.nely, cfg.nelx) >= 0.5;
visited = false(size(solid));
comp_grid = zeros(size(solid));
component_count = 0;
sizes = [];
touches_left = [];
touches_right = [];
for r = 1:size(solid,1)
    for col = 1:size(solid,2)
        if solid(r,col) && ~visited(r,col)
            component_count = component_count + 1;
            [pixels, left, right, visited] = flood_pixels(solid, visited, r, col);
            for k = 1:size(pixels,1)
                comp_grid(pixels(k,1), pixels(k,2)) = component_count;
            end
            sizes(component_count,1) = size(pixels,1); %#ok<AGROW>
            touches_left(component_count,1) = left; %#ok<AGROW>
            touches_right(component_count,1) = right; %#ok<AGROW>
        end
    end
end
comp_id = comp_grid(:);
total_solid = sum(sizes);
if isempty(sizes)
    largest_id = NaN; largest_fraction = NaN;
else
    [~, largest_id] = max(sizes);
    largest_fraction = sizes(largest_id) / max(total_solid, eps);
end
support_components = touches_left & touches_right;
c = struct();
c.component_count = component_count;
c.component_sizes = sizes;
c.component_touches_left = touches_left;
c.component_touches_right = touches_right;
c.connected = component_count == 1;
c.support_connected = any(support_components);
c.largest_component_id = largest_id;
c.largest_component_fraction = largest_fraction;
c.isolated_solid_fraction = sum(sizes(~support_components)) / max(total_solid, eps);
end

function [pixels, touches_left, touches_right, visited] = flood_pixels(solid, visited, r0, c0)
nr = size(solid,1); nc = size(solid,2);
queue = zeros(numel(solid),2); pixels = zeros(numel(solid),2);
head = 1; tail = 1; count = 0;
queue(tail,:) = [r0,c0]; visited(r0,c0) = true;
touches_left = false; touches_right = false;
while head <= tail
    r = queue(head,1); c = queue(head,2); head = head + 1;
    count = count + 1; pixels(count,:) = [r,c];
    touches_left = touches_left || c == 1;
    touches_right = touches_right || c == nc;
    neigh = [r-1 c; r+1 c; r c-1; r c+1];
    for k = 1:4
        rr = neigh(k,1); cc = neigh(k,2);
        if rr >= 1 && rr <= nr && cc >= 1 && cc <= nc && solid(rr,cc) && ~visited(rr,cc)
            tail = tail + 1; queue(tail,:) = [rr,cc]; visited(rr,cc) = true;
        end
    end
end
pixels = pixels(1:count,:);
end

function write_component_table(audit, path)
c = audit.connectedness;
ids = (1:c.component_count)';
if isempty(ids)
    T = table();
else
    T = table(ids, c.component_sizes(:), c.component_touches_left(:), ...
        c.component_touches_right(:), 'VariableNames', ...
        {'component','solid_elements','touches_left','touches_right'});
end
writetable(T, path);
end

function write_mode_energy_table(audit, path)
rows = {};
for j = 1:size(audit.total_fraction,2)
    for c = 1:size(audit.total_fraction,1)
        rows(end+1,:) = {j, c, audit.kinetic_fraction(c,j), audit.strain_fraction(c,j), ...
            audit.total_fraction(c,j)}; %#ok<AGROW>
    end
end
if isempty(rows)
    T = table();
else
    T = cell2table(rows, 'VariableNames', {'mode','component','kinetic_fraction', ...
        'strain_fraction','mean_energy_fraction'});
end
writetable(T, path);
end

function write_iteration_table(hist, path)
ni = hist.outer_iters;
omega = hist.omega_trial(1:ni,:);
gap12 = abs(omega(:,2) - omega(:,1)) ./ max(omega(:,1), eps);
T = table((1:ni)', omega(:,1), omega(:,2), gap12, hist.N_trial(1:ni), ...
    hist.volume(1:ni), hist.drho_norm(1:ni), hist.move_lim_effective(1:ni), ...
    hist.outer_move_effective(1:ni), 'VariableNames', ...
    {'iter','omega1','omega2','gap12','N_trial','volume','drho_norm', ...
    'move_lim_effective','outer_move_effective'});
writetable(T, path);
end

function T = summaries_to_table(s)
n = numel(s);
tag = cell(n,1); desc = cell(n,1); cls = cell(n,1);
omega1 = nan(n,1); omega2 = nan(n,1); gap12 = nan(n,1); N = nan(n,1);
paper_close = false(n,1); coalesced = false(n,1); support = false(n,1);
success = false(n,1); vol = nan(n,1); comps = nan(n,1); isolated = nan(n,1);
mode1sup = nan(n,1); mode2sup = nan(n,1); topology = cell(n,1);
for i = 1:n
    tag{i} = s(i).tag; desc{i} = s(i).description; cls{i} = s(i).classification;
    omega1(i) = s(i).omega1; omega2(i) = s(i).omega2; gap12(i) = s(i).gap12;
    N(i) = s(i).N; paper_close(i) = s(i).paper_close; coalesced(i) = s(i).coalesced;
    support(i) = s(i).support_connected; success(i) = s(i).success; vol(i) = s(i).volume;
    comps(i) = s(i).component_count; isolated(i) = s(i).isolated_fraction;
    mode1sup(i) = s(i).mode1_support_energy; mode2sup(i) = s(i).mode2_support_energy;
    topology{i} = s(i).topology_png;
end
T = table(tag, desc, omega1, omega2, gap12, N, paper_close, coalesced, ...
    support, success, cls, vol, comps, isolated, mode1sup, mode2sup, topology);
end

function export_topology(rho, cfg, summary, path)
fig = figure('Visible','off','Color','w');
imagesc(1 - reshape(rho, cfg.nely, cfg.nelx));
colormap(gray); axis equal tight off;
title(sprintf('%s | %.1f %.1f | support %d', strrep(summary.tag, '_', '\_'), ...
    summary.omega1, summary.omega2, summary.support_connected), 'Interpreter','tex');
exportgraphics(fig, path, 'Resolution', 180);
close(fig);
end

function write_report(path, s, paper_omega, paper_tol, gap_tol, support_energy_tol)
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# Missing Regularization / Benchmark-Assumption Diagnostics\n\n');
fprintf(fid, 'Generated: `%s`\n\n', datestr(now, 31));
fprintf(fid, ['Success guard: support-connected topology, `omega1 = %.1f +/- %.1f%%`, ', ...
    '`gap12 <= %.4g`, `N >= 2`, and support-connected component carrying at least ', ...
    '%.0f%% mean kinetic/strain energy in modes 1 and 2.\n\n'], ...
    paper_omega, 100*paper_tol, gap_tol, 100*support_energy_tol);
fprintf(fid, '## Summary\n\n');
fprintf(fid, ['| variant | omega1 | omega2 | gap | N | support | mode1/2 support energy | ', ...
    'components | isolated frac | success | classification |\n']);
fprintf(fid, '|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|\n');
for i = 1:numel(s)
    fprintf(fid, '| `%s` | %.3f | %.3f | %.4g | %g | %s | %.3f / %.3f | %d | %.3f | %s | %s |\n', ...
        s(i).tag, s(i).omega1, s(i).omega2, s(i).gap12, s(i).N, ...
        yesno(s(i).support_connected), s(i).mode1_support_energy, ...
        s(i).mode2_support_energy, s(i).component_count, s(i).isolated_fraction, ...
        yesno(s(i).success), s(i).classification);
end
fprintf(fid, '\n## Findings\n\n');
if any([s.success])
    fprintf(fid, '- Successful support-connected near-paper variants exist, but their classification depends on the added assumption shown in the table.\n');
else
    fprintf(fid, '- No tested diagnostic variant recovered a support-connected bimodal CC topology near `omega = %.1f` under the success guard.\n', paper_omega);
end
fprintf(fid, '- Variants with passive support-path material are connectivity diagnostics only and are not paper-faithful constraints.\n');
fprintf(fid, '- Density filtering, Heaviside projection, lumped mass, and altered support placement are treated as extra or ambiguous benchmark assumptions, not confirmed Du-Olhoff 2007 reproduction details.\n');
fprintf(fid, '\n## Answer\n\n');
success_idx = find([s.success]);
if isempty(success_idx)
    fprintf(fid, ['Within this controlled matrix, the missing ingredient was not identified as a simple density filter, ', ...
        'projection continuation, larger filter radius, lumped mass, tested support-placement alternative, ', ...
        'or minimum support-path heuristic. The variants split into two failure modes: unconstrained ', ...
        'filter/projection/mass/support alternatives remain coalesced but disconnected, while explicit ', ...
        'support-path or weak-support variants can be support-connected structural designs but lose ', ...
        'the bimodal near-456 target.\n']);
else
    fprintf(fid, 'The variants passing the guard are:\n');
    for k = success_idx
        fprintf(fid, '- `%s`: `%s`.\n', s(k).tag, s(k).classification);
    end
    fprintf(fid, ['No successful variant with an added filter/projection/connectivity/support/mass assumption should be called ', ...
        'paper-faithful unless that assumption is independently verified in Du & Olhoff 2007.\n']);
end
fprintf(fid, '\n## Evidence Files\n\n');
fprintf(fid, '- `missing_regularization_diagnostics_results/missing_regularization_diagnostics_summary.csv`\n');
fprintf(fid, '- `missing_regularization_diagnostics_results/<variant>/<variant>_components.csv`\n');
fprintf(fid, '- `missing_regularization_diagnostics_results/<variant>/<variant>_mode_component_energy.csv`\n');
fprintf(fid, '- `missing_regularization_diagnostics_results/<variant>/<variant>_topology.png`\n');
fprintf(fid, '- `missing_regularization_diagnostics_results/<variant>/<variant>_audit.mat`\n');
end

function s = yesno(v)
if v, s = 'yes'; else, s = 'no'; end
end
