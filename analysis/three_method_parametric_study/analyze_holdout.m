function analyze_holdout()
%ANALYZE_HOLDOUT  WP17/WP18 cross-resolution validation table.
%
%   No profile parameter is touched here.  The frozen profiles are replayed on
%   the hold-out meshes and scored with the same common E1/E2/E3 evaluators as
%   Stage A.  Robustness classes come from study_preregistration.json.

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
study = fullfile(repo,'analysis','three_method_parametric_study');
addpath(study);
out = fullfile(study,'results');

rows = {};
for mode = {'prospective','observer'}
    d = fullfile(study,'raw',mode{1});
    if ~exist(d,'dir'), continue; end
    f = dir(fullfile(d,'*.mat'));
    for i = 1:numel(f)
        S = load(fullfile(f(i).folder,f(i).name)); r = S.record;
        row = struct();
        row.run_id = r.run_id; row.profile = ''; row.method = r.method;
        row.mode = mode{1}; row.status = r.status;
        row.nelx = r.nelx; row.nely = r.nely; row.n_elements = r.nelx*r.nely;
        if isfield(r,'profile'), row.profile = r.profile; end
        row.n_iter_observed = NaN; row.practical_stop_iter = NaN;
        row.n_stage1 = NaN; row.n_stage2 = NaN;
        row.wall_time_s = NaN; row.loop_time_s = NaN;
        row.loop_time_practical_s = NaN; row.loop_time_per_iter_s = NaN;
        row.omega1_native = NaN; row.omega2_native = NaN; row.omega3_native = NaN;
        row.gap12_native = NaN; row.N_final = NaN; row.n_solver_failures = NaN;
        row.break_reason = '';
        if ~strcmp(r.status,'COMPLETED')
            row = padded(row);
            row.convergence_status = 'SOLVER_FAILURE';
            rows{end+1,1} = row; %#ok<AGROW>
            continue;
        end
        row.n_iter_observed = r.n_iter;
        row.practical_stop_iter = r.practical_stop_iter;
        row.wall_time_s = r.wall_time; row.loop_time_s = r.loop_time;
        row.loop_time_practical_s = r.loop_time_practical;
        row.loop_time_per_iter_s = r.loop_time/max(r.n_iter,1);
        row.omega1_native = r.omega_native(1);
        row.omega2_native = r.omega_native(2);
        row.omega3_native = r.omega_native(3);
        if isfield(r,'n_stage') && isstruct(r.n_stage)
            row.n_stage1 = r.n_stage.stage1; row.n_stage2 = r.n_stage.stage2;
        end
        if strcmp(r.method,'Olhoff')
            n = r.n_iter;
            row.gap12_native = r.telemetry.gaps_rel(1,n);
            row.N_final = r.hist.N(n);
            row.n_solver_failures = sum(~(r.hist.innerConv(1:n) & r.telemetry.lp_flag(1:n)==1 & ...
                r.telemetry.eig_ok(1:n) & r.telemetry.finite_ok(1:n)));
            row.break_reason = r.break_reason;
            volfrac = r.cfg.volfrac; nelx = r.cfg.nelx; nely = r.cfg.nely;
            xTerm = double(r.telemetry.rho_snapshots(:,n+1));
        else
            row.gap12_native = abs(r.omega_native(2)-r.omega_native(1))/max(r.omega_native(1),eps);
            row.n_solver_failures = getdef(r.telemetry.stopping,'n_subproblem_failures',0);
            row.break_reason = getdef(r.telemetry.stopping,'stop_reason','');
            volfrac = r.cfg.optimization.volume_fraction;
            nelx = r.cfg.domain.mesh.nelx; nely = r.cfg.domain.mesh.nely;
            xTerm = double(r.x_late(:));
        end
        row = attach(row, double(r.x_practical), 'practical', nelx, nely, volfrac);
        row = attach(row, xTerm, 'terminal', nelx, nely, volfrac);
        row.convergence_status = classify(row, r.method);
        rows{end+1,1} = row; %#ok<AGROW>
        fprintf('scored %s [%s]\n',r.run_id,mode{1});
    end
end
T = struct2table(cell2mat(rows));
writetable(T, fullfile(out,'cross_resolution_validation.csv'));
fprintf('wrote cross_resolution_validation.csv (%d rows)\n',height(T));
end

function row = padded(row)
row = attach(row,[],'practical',240,30,0.5);
row = attach(row,[],'terminal',240,30,0.5);
end

function row = attach(row,x,tag,nelx,nely,volfrac)
if isempty(x) || numel(x) ~= nelx*nely
    for m = {'E1','E2','E3'}
        for rep = {'raw','binary'}
            for j = 1:3
                row.(sprintf('omega%d_common_%s_%s_%s',j,rep{1},m{1},tag)) = NaN;
            end
        end
    end
    row.(['volume_' tag]) = NaN; row.(['vol_resid_' tag]) = NaN;
    row.(['grayness_' tag]) = NaN;
    row.(['connected_raw_' tag]) = NaN; row.(['connected_bin_' tag]) = NaN;
    row.(['largest_comp_frac_' tag]) = NaN;
    return;
end
ev = study_evaluate_design(x,nelx,nely,volfrac);
for m = {'E1','E2','E3'}
    wr = ev.(['omega_raw_' m{1}]); wb = ev.(['omega_binary_' m{1}]);
    for j = 1:3
        row.(sprintf('omega%d_common_raw_%s_%s',j,m{1},tag)) = wr(j);
        row.(sprintf('omega%d_common_binary_%s_%s',j,m{1},tag)) = wb(j);
    end
end
row.(['volume_' tag]) = ev.volume;
row.(['vol_resid_' tag]) = ev.volume_residual;
row.(['grayness_' tag]) = ev.grayness;
row.(['connected_raw_' tag]) = double(ev.connectivity_raw_05.left_right_connected);
row.(['connected_bin_' tag]) = double(ev.connectivity_binary.left_right_connected);
row.(['largest_comp_frac_' tag]) = ev.connectivity_raw_05.largest_component_fraction;
end

function s = classify(row, method)
if row.n_solver_failures > 0,             s = 'SOLVER_FAILURE'; return; end
if row.connected_raw_terminal == 0,       s = 'CONNECTIVITY_FAILURE'; return; end
if isnan(row.practical_stop_iter),        s = 'CAP_HIT'; return; end
if strcmp(method,'Olhoff')
    if row.gap12_native > 0.01,           s = 'STATIONARY_NOT_BIMODAL'; return; end
    s = 'CONVERGED_BIMODAL'; return;
end
s = 'CONVERGED_NATIVE';
end

function v = getdef(s,f,d)
if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
