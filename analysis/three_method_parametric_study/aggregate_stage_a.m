function aggregate_stage_a()
%AGGREGATE_STAGE_A  WP4/WP5/WP12 aggregation of every Stage A run.
%
%   Produces the run ledger and the three per-method result tables.  Every
%   attempted configuration appears, including the censored first-pass Yuksel
%   and Proposed runs, so no unfavourable run can drop out of the study.
%
%   Two states are evaluated for every run:
%     practical  -- the state the method's own practical stop rule selects
%                   (Olhoff: frozen H_balanced_v1 detector fire; Yuksel and
%                   Proposed: their native density-change test);
%     terminal   -- the last observed state, i.e. the continued trajectory.
%   Both are scored with the same common E1/E2/E3 raw and binary evaluators.
%   NOTE ON NAMING: the omega*_native_terminal columns are the solver's own
%   reported frequencies at the END of the observed trajectory, in that
%   method's own interpolation model.  They are not comparable across methods
%   and they are not the practical-stop values -- the practical-stop spectral
%   quality lives only in the omega*_common_*_practical columns.
%   Native and common quantities are kept in separate, explicitly named
%   columns and never mixed.

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
study = fullfile(repo,'analysis','three_method_parametric_study');
addpath(study);
addpath(fullfile(repo,'analysis','olhoff_native_convergence'));
out = fullfile(study,'results');
if ~exist(out,'dir'), mkdir(out); end

D = frozenDetector();
rows = {};

% ---------------- Olhoff ----------------------------------------------
f = dir(fullfile(study,'raw','stage_a','olhoff_move_*.mat'));
[~,o] = sort({f.name}); f = f(o);
for i = 1:numel(f)
    S = load(fullfile(f(i).folder,f(i).name)); r = S.record;
    rows{end+1,1} = olhoffRow(r,D,'stage_a'); %#ok<AGROW>
    fprintf('aggregated %s\n',r.run_id);
end

% ---------------- Yuksel and Proposed ---------------------------------
% stage_a is the censored first pass (300-iteration cap on both Yuksel stages
% and on the Proposed loop); stage_a_v2 is the uncensored re-run.
for pass = {'stage_a','stage_a_v2'}
    dirp = fullfile(study,'raw',pass{1});
    if ~exist(dirp,'dir'), continue; end
    g = dir(fullfile(dirp,'*.mat'));
    [~,o] = sort({g.name}); g = g(o);
    for i = 1:numel(g)
        if startsWith(g(i).name,'olhoff'), continue; end
        S = load(fullfile(g(i).folder,g(i).name)); r = S.record;
        rows{end+1,1} = ourRow(r,pass{1}); %#ok<AGROW>
        fprintf('aggregated %s [%s]\n',r.run_id,pass{1});
    end
end

T = struct2table(cell2mat(rows));
writetable(T, fullfile(out,'parametric_run_ledger.csv'));
writetable(T(strcmp(T.method,'Olhoff'),:),   fullfile(out,'olhoff_parametric_results.csv'));
writetable(T(strcmp(T.method,'Yuksel'),:),   fullfile(out,'yuksel_parametric_results.csv'));
writetable(T(strcmp(T.method,'Proposed'),:), fullfile(out,'proposed_parametric_results.csv'));
fprintf('wrote ledger with %d runs\n',height(T));
end

% =======================================================================
function d = frozenDetector()
% H_balanced_v1, replayed verbatim from the frozen move=0.005 audit.
% NOTHING here is retuned for this study: only the trajectory's move changes.
d = struct('objective_block',20,'window',40,'persistence',20, ...
    'objective_block_drift_tol',1e-4,'objective_phase_recurrence_tol',1e-4, ...
    'rho_phase_rms_tol',1.25e-3,'topology_phase_turnover_tol',7e-4, ...
    'modal_window',40,'gap_tol',1e-2,'volume_tol_rel',1e-8,'required_N',2);
end

function row = olhoffRow(r,D,pass)
row = baseRow(r,pass);
row.method = 'Olhoff';
row.family_params = sprintf('move=%g;rmin_el=1.3;lp;tolMult=%g',r.move,r.cfg.tolMult);
row.move = r.move; row.tol = r.cfg.tolOuter;
row.stage1_tol = NaN; row.stage2_tol = NaN;
row.n_stage1 = NaN; row.n_stage2 = NaN;
if ~strcmp(r.status,'COMPLETED_OBSERVER'), row = padEval(row); return; end

n = r.n_iter;
row.n_iter_terminal = n;
row.wall_time_terminal_s = r.wall_time;
% Optimization-loop time is the MEASURED per-iteration cost accumulated over
% the trajectory, not the total wall time divided by iterations.  Olhoff's
% per-iteration cost is strongly trajectory dependent (multiplicity handling
% makes later iterations dearer), so a uniform prorate would misprice any
% early stop.
loopCum = cumsum(r.hist.tEig(1:n)+r.hist.tGrad(1:n)+r.hist.tInner(1:n));
row.loop_time_terminal_s = loopCum(n);
row.t_iter_s = loopCum(n)/n;
row.eig_time_s = sum(r.hist.tEig(1:n));
row.eig_share = row.eig_time_s/max(loopCum(n),eps);
row.omega1_native_terminal = r.omega_native(1);
row.omega2_native_terminal = r.omega_native(2);
row.omega3_native_terminal = r.omega_native(3);
row.gap12_native_terminal = r.telemetry.gaps_rel(1,n);
row.N_final = r.hist.N(n);
row.native_rule_fires = double(any(r.hist.dxOuter(1:n) < r.cfg.tolOuter));
row.native_rule_iter = NaN;   % max|drho|<tolOuter never fires; see report

% Frozen detector replay, future-blind, evaluated at every k.
fire = NaN;
% The detector's phase-recurrence term reads omega(ix-2) over a 40-wide
% window, so the earliest iteration at which it is defined is
% window + 2 + persistence - 1 = 61.  Scanning from there is not a tolerance
% change: the configuration's own guards already make any fire before
% iteration 59 impossible, and every fire observed in this study is >= 154.
for k = 61:n
    if nativeConvergenceDetector(r.hist,r.telemetry,k,r.cfg,D), fire = k; break; end
end
row.practical_stop_iter = fire;
if isnan(fire)
    row.practical_status = 'CAP_HIT';
    row.wall_time_practical_s = NaN;
    xPract = [];
else
    row.practical_status = 'PRACTICAL_STOP';
    row.loop_time_practical_s = loopCum(fire);
    row.wall_time_practical_s = r.wall_time - (loopCum(n)-loopCum(fire));
    xPract = double(r.telemetry.rho_snapshots(:,fire+1));
end
row = attachEval(row, xPract, double(r.telemetry.rho_snapshots(:,n+1)), r.cfg.nelx, r.cfg.nely, r.cfg.volfrac);
row.n_solver_failures = sum(~(r.hist.innerConv(1:n) & r.telemetry.lp_flag(1:n)==1 & ...
    r.telemetry.eig_ok(1:n) & r.telemetry.finite_ok(1:n)));
row.validity = olhoffValidity(row);
end

function row = ourRow(r,pass)
row = baseRow(r,pass);
row.method = r.method;
row.move = NaN; row.tol = NaN; row.stage1_tol = NaN; row.stage2_tol = NaN;
if isfield(r,'move'), row.move = r.move; end
if isfield(r,'tol'), row.tol = r.tol; end
if isfield(r,'stage1_tol'), row.stage1_tol = r.stage1_tol; end
if isfield(r,'stage2_tol'), row.stage2_tol = r.stage2_tol; end
if strcmp(r.method,'Yuksel')
    row.tol = row.stage2_tol;
    row.family_params = sprintf('move=%g;s1tol=%g;s2tol=%g;rmin_el=2.5',row.move,row.stage1_tol,row.stage2_tol);
else
    row.family_params = sprintf('move=%g;tol=%g;rmin_el=2.0;OC',row.move,row.tol);
end
row.n_stage1 = NaN; row.n_stage2 = NaN;
if ~strcmp(r.status,'COMPLETED_OBSERVER'), row = padEval(row); return; end
if isstruct(r.n_stage)
    row.n_stage1 = r.n_stage.stage1; row.n_stage2 = r.n_stage.stage2;
end

n = r.n_iter;
row.n_iter_terminal = n;
row.wall_time_terminal_s = r.wall_time;
row.loop_time_terminal_s = r.loop_time;
row.t_iter_s = r.loop_time/n;
tm = r.telemetry.timing;
row.eig_time_s = getdef(tm,'eigensolve_time',NaN);
row.eig_share = row.eig_time_s/max(r.loop_time,eps);
row.omega1_native_terminal = r.omega_native(1);
row.omega2_native_terminal = r.omega_native(2);
row.omega3_native_terminal = r.omega_native(3);
row.gap12_native_terminal = abs(r.omega_native(2)-r.omega_native(1))/max(r.omega_native(1),eps);
row.N_final = NaN;
row.native_rule_fires = double(~isnan(r.native_stop_iter));
row.native_rule_iter = r.native_stop_iter;
row.practical_stop_iter = r.native_stop_iter;

xPract = [];
if ~isnan(r.native_stop_iter) && ~isempty(r.x_native)
    row.practical_status = 'PRACTICAL_STOP';
    xPract = double(r.x_native(:));
    % Measured cumulative loop time at the native stop iteration.  Yuksel's
    % two stages have different per-iteration costs (stage 1 solves no
    % eigenproblem), so a uniform prorate across the global iteration index
    % would misprice every stage-2 stop.
    row.loop_time_practical_s = elapsedAt(r.telemetry, r.native_stop_iter, r.loop_time, n);
    row.wall_time_practical_s = r.wall_time - (r.loop_time - row.loop_time_practical_s);
else
    row.practical_status = 'CAP_HIT';
    row.loop_time_practical_s = NaN;
    row.wall_time_practical_s = NaN;
end
nelx = r.cfg.domain.mesh.nelx; nely = r.cfg.domain.mesh.nely;
row = attachEval(row, xPract, double(r.x_late(:)), nelx, nely, r.cfg.optimization.volume_fraction);
row.n_solver_failures = getdef(r.telemetry.stopping,'n_subproblem_failures',0);
row.validity = ourValidity(row);
end

% -----------------------------------------------------------------------
function row = baseRow(r,pass)
row = struct();
row.run_id = r.run_id; row.pass = pass; row.method = ''; row.status = r.status;
row.nelx = 240; row.nely = 30;
if isfield(r,'nelx'), row.nelx = r.nelx; row.nely = r.nely; end
row.family_params = '';
row.move = NaN; row.tol = NaN; row.stage1_tol = NaN; row.stage2_tol = NaN;
row.n_stage1 = NaN; row.n_stage2 = NaN;
row.n_iter_terminal = NaN; row.wall_time_practical_s = NaN;
row.loop_time_practical_s = NaN; row.loop_time_terminal_s = NaN;
row.wall_time_terminal_s = NaN; row.t_iter_s = NaN;
row.eig_time_s = NaN; row.eig_share = NaN;
row.omega1_native_terminal = NaN; row.omega2_native_terminal = NaN;
row.omega3_native_terminal = NaN; row.gap12_native_terminal = NaN; row.N_final = NaN;
row.native_rule_fires = NaN; row.native_rule_iter = NaN;
row.practical_stop_iter = NaN; row.practical_status = 'NOT_RUN';
row.n_solver_failures = NaN; row.validity = 'UNKNOWN';
end

function row = padEval(row)
row = attachEval(row,[],[],240,30,0.5);
end

function row = attachEval(row,xPract,xTerm,nelx,nely,volfrac)
tags = {'practical','terminal'};
states = {xPract,xTerm};
for s = 1:2
    tag = tags{s}; x = states{s};
    if isempty(x)
        for m = {'E1','E2','E3'}
            for rep = {'raw','binary'}
                for j = 1:3
                    row.(sprintf('omega%d_common_%s_%s_%s',j,rep{1},m{1},tag)) = NaN;
                end
            end
        end
        row.(['volume_' tag]) = NaN; row.(['vol_resid_' tag]) = NaN;
        row.(['grayness_' tag]) = NaN; row.(['gray_frac_' tag]) = NaN;
        row.(['connected_raw_' tag]) = NaN; row.(['connected_bin_' tag]) = NaN;
        row.(['largest_comp_frac_' tag]) = NaN;
        continue;
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
    row.(['gray_frac_' tag]) = ev.gray_fraction_01_09;
    row.(['connected_raw_' tag]) = double(ev.connectivity_raw_05.left_right_connected);
    row.(['connected_bin_' tag]) = double(ev.connectivity_binary.left_right_connected);
    row.(['largest_comp_frac_' tag]) = ev.connectivity_raw_05.largest_component_fraction;
end
end

function v = olhoffValidity(row)
% WP7 taxonomy.  Failure precedence first, then modal validity, then whether
% a practical stop was actually reached.
if row.n_solver_failures > 0,               v = 'SOLVER_FAILURE'; return; end
if row.connected_raw_terminal == 0,         v = 'CONNECTIVITY_FAILURE'; return; end
if row.gap12_native_terminal > 0.01,                 v = 'STATIONARY_NOT_BIMODAL'; return; end
if isnan(row.practical_stop_iter),          v = 'CAP_HIT'; return; end
v = 'CONVERGED_BIMODAL';
end

function v = ourValidity(row)
if row.n_solver_failures > 0,               v = 'SOLVER_FAILURE'; return; end
if row.connected_raw_terminal == 0,         v = 'CONNECTIVITY_FAILURE'; return; end
if isnan(row.practical_stop_iter),          v = 'CAP_HIT'; return; end
% These methods do not target modal coalescence, so a wide eigengap is their
% expected outcome and is reported, not penalised.
v = 'CONVERGED_NATIVE';
end

function t = elapsedAt(tel, k, loopTime, n)
% Cumulative optimization-loop seconds at global iteration k, read from the
% per-iteration history when it is available.
t = loopTime*k/max(n,1);   % fallback only
if ~isfield(tel,'history') || ~isstruct(tel.history), return; end
h = tel.history;
if ~isfield(h,'iter') || ~isfield(h,'elapsed_s'), return; end
j = find(h.iter(:) == k, 1, 'first');
if ~isempty(j) && isfinite(h.elapsed_s(j)), t = h.elapsed_s(j); end
end

function v = getdef(s,f,d)
if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
