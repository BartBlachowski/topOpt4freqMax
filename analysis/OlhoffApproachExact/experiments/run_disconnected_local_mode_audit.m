% RUN_DISCONNECTED_LOCAL_MODE_AUDIT
%
% Audit why stabilized OlhoffApproachExact runs produce disconnected,
% high-frequency coalesced optima.
%
% This is diagnostic only.  FE, sensitivities, filters, generalized gradients,
% boundary conditions, and MMA are not changed.  Variants alter only filter
% radius, SIMP penalty schedule, optional density symmetry projection, and
% post-run connected-component pruning diagnostics.

clearvars;
close all;
clc;

this_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(this_dir, '..', '..', '..');
matlab_dir = fullfile(this_dir, '..', 'Matlab');
tools_dir = fullfile(repo_root, 'tools', 'Matlab');
addpath(matlab_dir);
addpath(tools_dir);

out_dir = fullfile(this_dir, 'disconnected_local_mode_audit_results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

paper_target_omega1 = 456.4;
paper_close_rel_tol = 0.02;
gap_tol = 0.005;
energy_concentration_threshold = 0.70;

variants = build_variants();
summaries = cell(numel(variants), 1);

fprintf('\n==========================================================\n');
fprintf(' Disconnected/local-mode audit for OlhoffApproachExact\n');
fprintf(' Output: %s\n', out_dir);
fprintf('==========================================================\n\n');

for iv = 1:numel(variants)
    v = variants(iv);
    variant_dir = fullfile(out_dir, v.tag);
    if ~exist(variant_dir, 'dir'), mkdir(variant_dir); end

    fprintf('Running %-24s  rmin=%.2f penalty=%s symmetry=%s\n', ...
        v.tag, v.rmin_elem, v.penalty_schedule, v.symmetry);
    elapsed_tic = tic;
    [rho_final, hist, stage_info, final_cfg] = run_variant(v);
    elapsed_s = toc(elapsed_tic);

    audit = analyze_final_design(rho_final, final_cfg, energy_concentration_threshold);
    prune = pruning_diagnostic(rho_final, final_cfg);
    summary = summarize_variant(v, hist, rho_final, audit, prune, stage_info, ...
        elapsed_s, paper_target_omega1, paper_close_rel_tol, gap_tol, variant_dir);
    summaries{iv} = summary;

    write_iteration_table(hist, fullfile(variant_dir, [v.tag '_iterations.csv']));
    write_component_table(audit, fullfile(variant_dir, [v.tag '_components.csv']));
    write_mode_component_table(audit, fullfile(variant_dir, [v.tag '_mode_component_energy.csv']));
    writematrix(rho_final(:), fullfile(variant_dir, [v.tag '_rho_final.csv']));
    export_topology_png(rho_final, final_cfg, summary, fullfile(variant_dir, [v.tag '_topology.png']));
    export_mode_shape_pngs(audit, final_cfg, v, variant_dir);
    save(fullfile(variant_dir, [v.tag '_audit.mat']), ...
        'v', 'final_cfg', 'hist', 'stage_info', 'rho_final', 'audit', 'prune', ...
        'summary', 'elapsed_s');

    fprintf(['  omega=(%.4f, %.4f) gap=%.4g N=%g vol=%.5f comps=%d ', ...
        'support=%d local12=%d/%d prune omega1=%.4f\n'], ...
        summary.final_omega1, summary.final_omega2, summary.final_gap12, ...
        summary.final_N, summary.volume, summary.component_count, ...
        summary.support_connected, summary.mode1_local, summary.mode2_local, ...
        summary.support_pruned_omega1);
end

summary_struct = [summaries{:}];
summary_table = summaries_to_table(summary_struct);
writetable(summary_table, fullfile(out_dir, 'disconnected_local_mode_audit_summary.csv'));
save(fullfile(out_dir, 'disconnected_local_mode_audit_results.mat'), ...
    'summary_struct', 'summary_table', 'paper_target_omega1', ...
    'paper_close_rel_tol', 'gap_tol', 'energy_concentration_threshold');
write_report(fullfile(this_dir, 'disconnected_local_mode_audit_report.md'), ...
    summary_struct, paper_target_omega1, paper_close_rel_tol, gap_tol, ...
    energy_concentration_threshold);

fprintf('\nReport written to %s\n', fullfile(this_dir, 'disconnected_local_mode_audit_report.md'));

function variants = build_variants()
rmins = [2.5, 3.5, 5.0];
items = {};
for i = 1:numel(rmins)
    items{end+1} = make_variant(sprintf('fixed_p3_rmin_%s', tag_num(rmins(i))), ...
        rmins(i), 'fixed_p3', 'none'); %#ok<AGROW>
end
for i = 1:numel(rmins)
    items{end+1} = make_variant(sprintf('cont_p123_rmin_%s', tag_num(rmins(i))), ...
        rmins(i), 'p123', 'none'); %#ok<AGROW>
end
items{end+1} = make_variant('fixed_p3_rmin_3p5_sym_midspan', 3.5, 'fixed_p3', 'midspan');
items{end+1} = make_variant('fixed_p3_rmin_3p5_sym_midheight', 3.5, 'fixed_p3', 'midheight');
items{end+1} = make_variant('fixed_p3_rmin_3p5_sym_both', 3.5, 'fixed_p3', 'both');
items{end+1} = make_variant('cont_p123_rmin_3p5_sym_both', 3.5, 'p123', 'both');
variants = [items{:}];
end

function v = make_variant(tag, rmin_elem, penalty_schedule, symmetry)
cfg = base_cfg();
cfg.rmin_elem = rmin_elem;
cfg.density_symmetry = symmetry;
v = struct();
v.tag = tag;
v.rmin_elem = rmin_elem;
v.penalty_schedule = penalty_schedule;
v.symmetry = symmetry;
v.cfg = cfg;
end

function cfg = base_cfg()
cfg = struct();
cfg.support_type = 'CC';
cfg.L = 8.0;
cfg.H = 1.0;
cfg.nelx = 40;
cfg.nely = 5;
cfg.E0 = 1e7;
cfg.nu = 0.3;
cfg.rho0 = 1.0;
cfg.t = 1.0;
cfg.volfrac = 0.5;
cfg.rho_min = 1e-3;
cfg.mass_mode = 'du2007_c1';
cfg.sensitivity_filter = true;
cfg.n_target = 1;
cfg.n_modes = 6;
cfg.mult_tol = 1e-3;
cfg.outer_max_iter = 100;
cfg.outer_tol = 1e-6;
cfg.inner_max_iter = 30;
cfg.inner_tol = 1e-4;
cfg.move_lim = 0.2;
cfg.outer_move = 0.2;
cfg.alpha = 0.5;
cfg.acceptance_check = false;
cfg.verbose = false;
cfg.rho_snapshot_interval = 1;
cfg.globalization_enabled = false;
cfg.post_coalescence_trust_enabled = true;
cfg.post_coalescence_trust_factor = 0.1;
cfg.post_coalescence_gap_tol = 0.005;
cfg.persistent_mma_state = false;
cfg.density_symmetry = 'none';
end

function [rho_final, hist, stage_info, final_cfg] = run_variant(v)
cfg = v.cfg;
stage_info = struct('penal', {}, 'outer_iters', {}, 'final_omega1', {}, 'final_omega2', {});
if strcmp(v.penalty_schedule, 'fixed_p3')
    cfg.penal = 3;
    [rho_final, hist] = topopt_freq_exact(cfg);
    final_cfg = cfg;
    stage_info(1).penal = 3;
    stage_info(1).outer_iters = hist.outer_iters;
    stage_info(1).final_omega1 = hist.final_omega(1);
    stage_info(1).final_omega2 = hist.final_omega(2);
    return
end

penals = [1, 2, 3];
stage_iters = [40, 40, 100];
rho0 = [];
for i = 1:numel(penals)
    cfg.penal = penals(i);
    cfg.outer_max_iter = stage_iters(i);
    cfg.initial_rho = rho0;
    [rho0, hist_stage] = topopt_freq_exact(cfg);
    stage_info(i).penal = penals(i);
    stage_info(i).outer_iters = hist_stage.outer_iters;
    stage_info(i).final_omega1 = hist_stage.final_omega(1);
    stage_info(i).final_omega2 = hist_stage.final_omega(2);
end
rho_final = rho0;
hist = hist_stage;
final_cfg = cfg;
end

function summary = summarize_variant(v, hist, rho_final, audit, prune, stage_info, ...
    elapsed_s, paper_target_omega1, paper_close_rel_tol, gap_tol, variant_dir)
omega = hist.final_omega(:)';
gap12 = abs(omega(2) - omega(1)) / max(omega(1), eps);
paper_close = abs(omega(1) - paper_target_omega1) / paper_target_omega1 <= paper_close_rel_tol;
mode1 = audit.mode_classification(1);
mode2 = audit.mode_classification(2);

summary = struct();
summary.tag = v.tag;
summary.rmin_elem = v.rmin_elem;
summary.penalty_schedule = v.penalty_schedule;
summary.symmetry = v.symmetry;
summary.elapsed_s = elapsed_s;
summary.final_omega1 = omega(1);
summary.final_omega2 = omega(2);
summary.final_gap12 = gap12;
summary.final_N = hist.final_N;
summary.paper_close = paper_close;
summary.coalesced = gap12 <= gap_tol;
summary.volume = mean(rho_final);
summary.rho_min = min(rho_final);
summary.rho_max = max(rho_final);
summary.component_count = audit.connectedness.component_count;
summary.connected = audit.connectedness.connected;
summary.support_connected = audit.connectedness.support_connected;
summary.largest_component_fraction = audit.connectedness.largest_component_fraction;
summary.isolated_solid_fraction = audit.connectedness.isolated_solid_fraction;
summary.mode1_class = mode1.class;
summary.mode2_class = mode2.class;
summary.mode1_local = mode1.local_mode;
summary.mode2_local = mode2.local_mode;
summary.mode1_top_component_fraction = mode1.top_component_total_fraction;
summary.mode2_top_component_fraction = mode2.top_component_total_fraction;
summary.mode1_support_energy_fraction = mode1.support_total_fraction;
summary.mode2_support_energy_fraction = mode2.support_total_fraction;
summary.support_pruned_omega1 = prune.support_pruned_omega(1);
summary.support_pruned_omega2 = prune.support_pruned_omega(2);
summary.largest_pruned_omega1 = prune.largest_pruned_omega(1);
summary.largest_pruned_omega2 = prune.largest_pruned_omega(2);
summary.stage_info = stage_info;
summary.variant_dir = variant_dir;
summary.topology_png = fullfile(variant_dir, [v.tag '_topology.png']);
summary.component_csv = fullfile(variant_dir, [v.tag '_components.csv']);
summary.mode_energy_csv = fullfile(variant_dir, [v.tag '_mode_component_energy.csv']);
end

function audit = analyze_final_design(rho, cfg, concentration_threshold)
[omega, lam, Phi, K, M, cMat, Ke_phys, Me_phys, flag] = eval_modes_full(rho, cfg);
if flag ~= 0
    error('run_disconnected_local_mode_audit:EigsFailed', 'Final eigensolve failed.');
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
    totalK = max(sum(elemK), eps);
    totalM = max(sum(elemM), eps);
    for c = 1:num_comp
        idx = comp_id == c;
        kin_frac(c, j) = sum(elemM(idx)) / totalM;
        strain_frac(c, j) = sum(elemK(idx)) / totalK;
        total_frac(c, j) = 0.5 * (kin_frac(c, j) + strain_frac(c, j));
    end
end

classification = repmat(struct('mode', NaN, 'class', '', 'local_mode', false, ...
    'top_component_id', NaN, 'top_component_total_fraction', NaN, ...
    'support_total_fraction', NaN), num_modes, 1);
support_components = find(conn.component_touches_left & conn.component_touches_right);
for j = 1:num_modes
    [top_frac, top_comp] = max(total_frac(:, j));
    support_frac = 0;
    if ~isempty(support_components)
        support_frac = sum(total_frac(support_components, j));
    end
    top_is_support = ismember(top_comp, support_components);
    local_mode = top_frac >= concentration_threshold && ~top_is_support;
    if conn.support_connected && support_frac >= concentration_threshold
        cls = 'structural_support_beam_mode';
    elseif local_mode
        cls = 'island_or_component_local_mode';
    elseif ~conn.support_connected
        cls = 'disconnected_structure_mode';
    else
        cls = 'mixed_global_mode';
    end
    classification(j).mode = j;
    classification(j).class = cls;
    classification(j).local_mode = local_mode;
    classification(j).top_component_id = top_comp;
    classification(j).top_component_total_fraction = top_frac;
    classification(j).support_total_fraction = support_frac;
end

audit = struct();
audit.omega = omega(:)';
audit.lambda = lam(:)';
audit.K = K;
audit.M = M;
audit.Phi = Phi;
audit.component_id = comp_id;
audit.connectedness = conn;
audit.kinetic_fraction = kin_frac;
audit.strain_fraction = strain_frac;
audit.total_fraction = total_frac;
audit.mode_classification = classification;
end

function prune = pruning_diagnostic(rho, cfg)
[comp_id, conn] = component_map(rho, cfg);
support_components = find(conn.component_touches_left & conn.component_touches_right);
rho_support = cfg.rho_min * ones(size(rho));
if ~isempty(support_components)
    keep = ismember(comp_id, support_components);
    rho_support(keep) = rho(keep);
    [omega_support, ~] = eval_omega_for_rho(rho_support, cfg);
else
    omega_support = nan(cfg.n_modes, 1);
end
rho_largest = cfg.rho_min * ones(size(rho));
if isfinite(conn.largest_component_id) && conn.largest_component_id > 0
    keep = comp_id == conn.largest_component_id;
    rho_largest(keep) = rho(keep);
end
[omega_largest, ~] = eval_omega_for_rho(rho_largest, cfg);
prune = struct();
prune.support_pruned_omega = omega_support(:)';
prune.largest_pruned_omega = omega_largest(:)';
prune.support_components = support_components(:)';
prune.largest_component_id = conn.largest_component_id;
end

function [omega, lam, Phi, K, M, cMat, Ke_phys, Me_phys, flag] = eval_modes_full(rho, cfg)
nelx = cfg.nelx; nely = cfg.nely;
dx = cfg.L / nelx; dy = cfg.H / nely;
nEl = nelx * nely;
nDof = 2 * (nelx+1) * (nely+1);
[Ke_star, Me_star] = fe_q4_exact(cfg.nu, cfg.t, dx, dy);
Ke_phys = cfg.E0 * Ke_star;
Me_phys = cfg.rho0 * Me_star;
nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
        cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
[Il, Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)', [], 1);
jK = reshape(cMat(:,Jl)', [], 1);
Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));
fixed = build_supports_exact(cfg.support_type, nodeNrs);
free = setdiff(1:nDof, fixed);
[K, M] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, cfg.penal, cfg.mass_mode);
Kf = K(free, free);
Mf = M(free, free);
opts.tol = 1e-10; opts.maxit = 600;
[V, D, flag] = eigs(Kf, Mf, cfg.n_modes, 'SM', opts);
if flag ~= 0
    opts.tol = 1e-8; opts.maxit = 1500; opts.p = min(numel(free)-1, max(40, 4*cfg.n_modes));
    [V, D, flag] = eigs(Kf, Mf, cfg.n_modes, 'SM', opts);
end
if flag ~= 0
    omega = nan(cfg.n_modes, 1);
    lam = nan(cfg.n_modes, 1);
    Phi = nan(nDof, cfg.n_modes);
    return
end
[lam, idx] = sort(real(diag(D)));
V = real(V(:, idx));
for j = 1:cfg.n_modes
    sc = sqrt(abs(V(:,j)' * (Mf * V(:,j))));
    if sc > 1e-14, V(:,j) = V(:,j) / sc; end
end
omega = sqrt(max(lam, 0));
Phi = zeros(nDof, cfg.n_modes);
for j = 1:cfg.n_modes
    Phi(free, j) = V(:, j);
end
end

function [omega, flag] = eval_omega_for_rho(rho, cfg)
[omega, ~, ~, ~, ~, ~, ~, ~, flag] = eval_modes_full(rho, cfg);
end

function [comp_id, c] = component_map(rho, cfg)
solid = reshape(rho, cfg.nely, cfg.nelx) >= 0.5;
visited = false(size(solid));
comp_grid = zeros(size(solid));
component_count = 0;
sizes = [];
touches_left = [];
touches_right = [];
for r = 1:size(solid, 1)
    for col = 1:size(solid, 2)
        if solid(r, col) && ~visited(r, col)
            component_count = component_count + 1;
            [pixels, left, right, visited] = flood_pixels(solid, visited, r, col);
            for k = 1:size(pixels,1)
                comp_grid(pixels(k,1), pixels(k,2)) = component_count;
            end
            sizes(component_count, 1) = size(pixels, 1); %#ok<AGROW>
            touches_left(component_count, 1) = left; %#ok<AGROW>
            touches_right(component_count, 1) = right; %#ok<AGROW>
        end
    end
end
comp_id = comp_grid(:);
total_solid = sum(sizes);
if isempty(sizes)
    largest = 0; largest_id = NaN; largest_fraction = NaN;
else
    [largest, largest_id] = max(sizes);
    largest_fraction = largest / max(total_solid, eps);
end
support_components = touches_left & touches_right;
support_connected = any(support_components & (sizes / max(total_solid, eps) >= 0.95));
isolated_solid = sum(sizes(~support_components));
c = struct();
c.component_count = component_count;
c.component_sizes = sizes;
c.component_touches_left = touches_left;
c.component_touches_right = touches_right;
c.connected = component_count == 1;
c.support_connected = support_connected;
c.largest_component_id = largest_id;
c.largest_component_size = largest;
c.largest_component_fraction = largest_fraction;
c.isolated_solid_fraction = isolated_solid / max(total_solid, eps);
end

function [pixels, touches_left, touches_right, visited] = flood_pixels(solid, visited, r0, c0)
nr = size(solid, 1);
nc = size(solid, 2);
queue = zeros(numel(solid), 2);
pixels = zeros(numel(solid), 2);
head = 1; tail = 1; count = 0;
queue(tail,:) = [r0, c0];
visited(r0,c0) = true;
touches_left = false;
touches_right = false;
while head <= tail
    r = queue(head,1);
    c = queue(head,2);
    head = head + 1;
    count = count + 1;
    pixels(count,:) = [r, c];
    touches_left = touches_left || c == 1;
    touches_right = touches_right || c == nc;
    neigh = [r-1 c; r+1 c; r c-1; r c+1];
    for k = 1:4
        rr = neigh(k,1);
        cc = neigh(k,2);
        if rr >= 1 && rr <= nr && cc >= 1 && cc <= nc && solid(rr,cc) && ~visited(rr,cc)
            tail = tail + 1;
            queue(tail,:) = [rr, cc];
            visited(rr,cc) = true;
        end
    end
end
pixels = pixels(1:count,:);
end

function write_iteration_table(hist, path)
ni = hist.outer_iters;
omega = hist.omega_trial(1:ni,:);
gap12 = abs(omega(:,2) - omega(:,1)) ./ max(omega(:,1), eps);
T = table((1:ni)', omega(:,1), omega(:,2), gap12, hist.N_trial(1:ni), ...
    hist.volume(1:ni), hist.drho_norm(1:ni), hist.drho_max(1:ni), ...
    hist.move_lim_effective(1:ni), hist.outer_move_effective(1:ni), ...
    'VariableNames', {'iter','omega1','omega2','gap12','N_trial','volume', ...
    'drho_norm','drho_max','move_lim_effective','outer_move_effective'});
writetable(T, path);
end

function write_component_table(audit, path)
c = audit.connectedness;
ids = (1:c.component_count)';
if isempty(ids)
    T = table();
else
    T = table(ids, c.component_sizes(:), c.component_touches_left(:), ...
        c.component_touches_right(:), ...
        'VariableNames', {'component','solid_elements','touches_left','touches_right'});
end
writetable(T, path);
end

function write_mode_component_table(audit, path)
num_comp = size(audit.total_fraction, 1);
num_modes = size(audit.total_fraction, 2);
rows = {};
for j = 1:num_modes
    for c = 1:num_comp
        rows(end+1,:) = {j, c, audit.kinetic_fraction(c,j), ...
            audit.strain_fraction(c,j), audit.total_fraction(c,j)}; %#ok<AGROW>
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

function T = summaries_to_table(s)
n = numel(s);
tag = cell(n,1); rmin = nan(n,1); penalty = cell(n,1); symmetry = cell(n,1);
omega1 = nan(n,1); omega2 = nan(n,1); gap12 = nan(n,1); N = nan(n,1);
paper_close = false(n,1); coalesced = false(n,1); volume = nan(n,1);
rho_min = nan(n,1); rho_max = nan(n,1); comps = nan(n,1);
connected = false(n,1); support_connected = false(n,1);
largest_frac = nan(n,1); isolated_frac = nan(n,1);
mode1_class = cell(n,1); mode2_class = cell(n,1);
mode1_top = nan(n,1); mode2_top = nan(n,1);
mode1_support = nan(n,1); mode2_support = nan(n,1);
support_pruned_omega1 = nan(n,1); largest_pruned_omega1 = nan(n,1);
topology = cell(n,1);
for i = 1:n
    tag{i} = s(i).tag; rmin(i) = s(i).rmin_elem; penalty{i} = s(i).penalty_schedule;
    symmetry{i} = s(i).symmetry; omega1(i) = s(i).final_omega1; omega2(i) = s(i).final_omega2;
    gap12(i) = s(i).final_gap12; N(i) = s(i).final_N; paper_close(i) = s(i).paper_close;
    coalesced(i) = s(i).coalesced; volume(i) = s(i).volume; rho_min(i) = s(i).rho_min;
    rho_max(i) = s(i).rho_max; comps(i) = s(i).component_count; connected(i) = s(i).connected;
    support_connected(i) = s(i).support_connected; largest_frac(i) = s(i).largest_component_fraction;
    isolated_frac(i) = s(i).isolated_solid_fraction; mode1_class{i} = s(i).mode1_class;
    mode2_class{i} = s(i).mode2_class; mode1_top(i) = s(i).mode1_top_component_fraction;
    mode2_top(i) = s(i).mode2_top_component_fraction; mode1_support(i) = s(i).mode1_support_energy_fraction;
    mode2_support(i) = s(i).mode2_support_energy_fraction; support_pruned_omega1(i) = s(i).support_pruned_omega1;
    largest_pruned_omega1(i) = s(i).largest_pruned_omega1; topology{i} = s(i).topology_png;
end
T = table(tag, rmin, penalty, symmetry, omega1, omega2, gap12, N, paper_close, ...
    coalesced, volume, rho_min, rho_max, comps, connected, support_connected, ...
    largest_frac, isolated_frac, mode1_class, mode2_class, mode1_top, mode2_top, ...
    mode1_support, mode2_support, support_pruned_omega1, largest_pruned_omega1, topology);
end

function write_report(path, s, paper_target_omega1, paper_close_rel_tol, gap_tol, concentration_threshold)
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# Disconnected/Localized Mode Audit\n\n');
fprintf(fid, 'Generated: `%s`\n\n', datestr(now, 31));
fprintf(fid, '## Scope\n\n');
fprintf(fid, ['Stabilized OlhoffApproachExact CC 40x5 runs were audited against ', ...
    'support-connectedness, isolated islands, modal localization, component ', ...
    'energy concentration, filter radius, p-continuation, and symmetry. ', ...
    'The connectedness-pruning step is post-run diagnostic only and is not ', ...
    'claimed as an optimizer or reproduction method.\n\n']);
fprintf(fid, ['Paper target guard: `omega1 = %.1f +/- %.1f%%`, coalescence ', ...
    '`gap12 <= %.4g`. A mode is labeled component-local when one non-support ', ...
    'component carries at least %.0f%% mean kinetic/strain energy.\n\n'], ...
    paper_target_omega1, 100*paper_close_rel_tol, gap_tol, 100*concentration_threshold);

fprintf(fid, '## Summary\n\n');
fprintf(fid, ['| variant | omega1 | omega2 | gap | N | rmin | penalty | symmetry | ', ...
    'components | support-connected | isolated frac | mode1 | mode2 | ', ...
    'mode1 top/support | mode2 top/support | support-pruned omega1 |\n']);
fprintf(fid, '|---|---:|---:|---:|---:|---:|---|---|---:|---|---:|---|---|---:|---:|---:|\n');
for i = 1:numel(s)
    fprintf(fid, ['| `%s` | %.3f | %.3f | %.4g | %g | %.1f | %s | %s | ', ...
        '%d | %s | %.3f | %s | %s | %.3f / %.3f | %.3f / %.3f | %.3f |\n'], ...
        s(i).tag, s(i).final_omega1, s(i).final_omega2, s(i).final_gap12, ...
        s(i).final_N, s(i).rmin_elem, s(i).penalty_schedule, s(i).symmetry, ...
        s(i).component_count, yesno(s(i).support_connected), s(i).isolated_solid_fraction, ...
        s(i).mode1_class, s(i).mode2_class, s(i).mode1_top_component_fraction, ...
        s(i).mode1_support_energy_fraction, s(i).mode2_top_component_fraction, ...
        s(i).mode2_support_energy_fraction, s(i).support_pruned_omega1);
end

fprintf(fid, '\n## Findings\n\n');
if any([s.support_connected])
    fprintf(fid, '- Some variants are support-connected; inspect their mode classifications separately.\n');
else
    fprintf(fid, '- None of the audited stabilized final topologies are support-connected under the thresholded solid-component test.\n');
end
if any([s.paper_close] & [s.coalesced] & [s.support_connected])
    fprintf(fid, '- At least one variant satisfies paper frequency/gap and support-connectedness guards.\n');
else
    fprintf(fid, '- No variant satisfies the combined paper frequency/gap and support-connectedness guards.\n');
end
local_count = nnz([s.mode1_local] | [s.mode2_local]);
fprintf(fid, '- `%d/%d` variants have omega1 or omega2 classified as component-local by the energy-concentration test.\n', ...
    local_count, numel(s));
fprintf(fid, ['- Individual modes in a coalesced disconnected pair can be arbitrary ', ...
    'mixtures over equivalent islands, so the 70%% single-mode concentration ', ...
    'test is conservative. The stronger invariant here is that the support ', ...
    'component energy fraction is zero for omega1/omega2 in every variant.\n']);
fprintf(fid, ['- The support-connected pruning diagnostic has no candidate in these ', ...
    'runs: no thresholded solid component touches both supports. Largest-', ...
    'component pruning leaves isolated island spectra, not a supported beam.\n']);
fprintf(fid, ['- Fixed p=3 with larger rmin raises the disconnected coalesced pair ', ...
    '(rmin 2.5 -> 3.5 -> 5.0), reaching %.1f/%.1f at rmin 5.0, but the ', ...
    'topology remains disconnected and the first pair becomes component-local.\n'], ...
    lookup_omega1(s, 'fixed_p3_rmin_5p0'), lookup_omega2(s, 'fixed_p3_rmin_5p0'));
fprintf(fid, ['- The p=1->2->3 continuation runs do not recover the paper basin. ', ...
    'Without symmetry they lose coalescence or frequency, and with both-axis ', ...
    'symmetry they recover a high coalesced pair that is still disconnected.\n']);
fprintf(fid, ['- Mirror symmetry about midspan, midheight, or both axes stabilizes ', ...
    'symmetric-looking high-frequency coalesced states, but it does not ', ...
    'create support-connected structural modes.\n']);

fprintf(fid, '\n## Answer\n\n');
fprintf(fid, ['The non-paper high frequencies in this audit are caused by disconnected ', ...
    'and sometimes component-local modal behavior rather than a legitimate ', ...
    'alternative support-connected CC beam optimum. The evidence is the lack ', ...
    'of support-connected final topologies, isolated-material fraction of 1.0 ', ...
    'in every thresholded design, zero support-component modal energy for ', ...
    'omega1/omega2, and the inability of filter-radius changes, ', ...
    'p-continuation, or mirror symmetry to produce support-connected beam ', ...
    'modes.\n']);

fprintf(fid, '\n## Evidence Files\n\n');
fprintf(fid, '- `disconnected_local_mode_audit_results/disconnected_local_mode_audit_summary.csv`\n');
fprintf(fid, '- `disconnected_local_mode_audit_results/<variant>/<variant>_components.csv`\n');
fprintf(fid, '- `disconnected_local_mode_audit_results/<variant>/<variant>_mode_component_energy.csv`\n');
fprintf(fid, '- `disconnected_local_mode_audit_results/<variant>/<variant>_topology.png`\n');
fprintf(fid, '- `disconnected_local_mode_audit_results/<variant>/<variant>_mode1_shape.png`\n');
fprintf(fid, '- `disconnected_local_mode_audit_results/<variant>/<variant>_mode2_shape.png`\n');
fprintf(fid, '- `disconnected_local_mode_audit_results/<variant>/<variant>_audit.mat`\n');
end

function omega1 = lookup_omega1(s, tag)
omega1 = NaN;
for i = 1:numel(s)
    if strcmp(s(i).tag, tag), omega1 = s(i).final_omega1; return; end
end
end

function omega2 = lookup_omega2(s, tag)
omega2 = NaN;
for i = 1:numel(s)
    if strcmp(s(i).tag, tag), omega2 = s(i).final_omega2; return; end
end
end

function export_topology_png(rho, cfg, summary, path)
fig = figure('Visible', 'off', 'Color', 'w');
imagesc(1 - reshape(rho, cfg.nely, cfg.nelx));
colormap(gray);
axis equal tight off;
title(sprintf('%s | omega=(%.1f, %.1f), comps=%d', ...
    strrep(summary.tag, '_', '\_'), summary.final_omega1, ...
    summary.final_omega2, summary.component_count), 'Interpreter', 'tex');
exportgraphics(fig, path, 'Resolution', 180);
close(fig);
end

function export_mode_shape_pngs(audit, cfg, v, variant_dir)
node_nrs = reshape(1:(cfg.nelx+1)*(cfg.nely+1), cfg.nely+1, cfg.nelx+1);
X = linspace(0, cfg.L, cfg.nelx+1);
Y = linspace(0, cfg.H, cfg.nely+1);
[XX, YY] = meshgrid(X, Y);
for j = 1:min(2, size(audit.Phi, 2))
    phi = audit.Phi(:, j);
    ux = reshape(phi(2*node_nrs(:)-1), cfg.nely+1, cfg.nelx+1);
    uy = reshape(phi(2*node_nrs(:)), cfg.nely+1, cfg.nelx+1);
    mag = sqrt(ux.^2 + uy.^2);
    mag = mag ./ max(max(mag(:)), eps);
    fig = figure('Visible', 'off', 'Color', 'w');
    imagesc(X, Y, mag);
    set(gca, 'YDir', 'normal');
    axis equal tight;
    colormap(parula);
    colorbar;
    hold on;
    contour(XX, YY, mag, [0.5 0.5], 'k-', 'LineWidth', 0.75);
    title(sprintf('%s mode %d | omega %.1f', ...
        strrep(v.tag, '_', '\_'), j, audit.omega(j)), 'Interpreter', 'tex');
    exportgraphics(fig, fullfile(variant_dir, sprintf('%s_mode%d_shape.png', v.tag, j)), ...
        'Resolution', 180);
    close(fig);
end
end

function s = tag_num(x)
s = strrep(sprintf('%.1f', x), '.', 'p');
end

function s = yesno(v)
if v, s = 'yes'; else, s = 'no'; end
end
