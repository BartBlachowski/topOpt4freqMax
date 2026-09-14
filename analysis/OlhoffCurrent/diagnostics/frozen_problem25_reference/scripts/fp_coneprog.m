function out = fp_coneprog()
%FP_CONEPROG  Part 7: primary reference solve of the certified SOCP with
%   MATLAB coneprog, plus an explicit weak-duality certificate and the KKT
%   check of the EXACT production problem at the returned point.
%   FROZEN-SUBPROBLEM REFERENCE -- drho is never applied to a density.
S = fp_setup();                                   %#ok<NASGU>
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'),'ctx'); P = fp_problem(L.ctx);
NE = P.NE; nvar = P.nvar;

soc = secondordercone(P.Ac, P.bc, P.dc, P.gammac);
opts = optimoptions('coneprog','Display','iter','OptimalityTolerance',1e-10, ...
    'ConstraintTolerance',1e-10,'MaxIterations',500,'LinearSolver','auto');
t0 = tic;
[x, fv, ef, op, lam] = coneprog(P.f, soc, P.Alin, P.blin, [], [], P.xmin, P.xmax, opts);
wall = toc(t0);
xRef = x;
o = struct('label','FROZEN-SUBPROBLEM REFERENCE (coneprog) -- drho discarded, never applied');
o.exitflag = ef; o.output = op; o.wall_s = wall; o.fval_solver = fv;
o.options = struct('OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10,'MaxIterations',500,'LinearSolver','auto');
o.bs = x(end); o.beta = x(end)*P.lamref; o.omega_pred = sqrt(o.beta);
o.bs_minus_1 = x(end) - 1;

% ---- duals as returned ------------------------------------------------
o.lambda_fields = fieldnames(lam).';
mu = lam.soc; if iscell(mu), mu = mu{1}; end
o.lambda_soc = mu(:).'; o.lambda_ineqlin = lam.ineqlin(:).';
xsi = lam.lower(:); eta = lam.upper(:);
o.lambda_lower_stats = [min(xsi) max(xsi) nnz(xsi > 1e-8*max(xsi))];
o.lambda_upper_stats = [min(eta) max(eta) nnz(eta > 1e-8*max(eta))];

% ---- cone geometry at the solution ------------------------------------
s = P.Ac*x - P.bc; ns = norm(s); rhs = P.dc.'*x - P.gammac;
o.cone = struct('norm_Acx_minus_bc',ns,'dcx_minus_gamma',rhs,'residual',ns-rhs, ...
    'separation_e2_minus_e1',2*ns*P.lamref,'separation_over_d2',2*ns*P.lamref/P.d2);
[fval, dfdx, dlam, ~, evals] = P.evalProd(x); %#ok<ASGLU>
o.production_fval = fval.'; o.evals = evals.';

% ---- weak-duality certificate (independent of solver status) ----------
% For mu>=0, nu>=0 and any ||w||<=1:  ||Ac x-bc|| >= w'(Ac x-bc), so
%   p* >= min_{box} [f + mu(Ac'w-dc) + Alin'nu]'x  - mu w'bc + mu gamma - nu'blin.
muS = max(mu(1),0); nu = max(lam.ineqlin(:),0); w = s/ns;
q = P.f + muS*(P.Ac.'*w - P.dc) + P.Alin.'*nu;
Dval = sum(min(q.*P.xmin, q.*P.xmax)) - muS*(w.'*P.bc) + muS*P.gammac - nu.'*P.blin;
o.duality = struct('primal',P.f.'*x,'dual_bound',Dval,'gap',P.f.'*x - Dval, ...
    'mu_used',muS,'nu_used',nu.','w_norm',norm(w),'w',w.', ...
    'mu_negative_clipped',mu(1) < 0,'nu_negative_clipped',any(lam.ineqlin < 0));
% implied box multipliers from stationarity, for comparison with the returned ones
xsiImp = max(q,0); etaImp = max(-q,0);
o.box_multiplier_compare = struct('max_abs_xsi_diff',max(abs(xsiImp-xsi)),'max_abs_eta_diff',max(abs(etaImp-eta)), ...
    'max_xsi',max(xsi),'max_eta',max(eta));

% ---- KKT of the exact production problem with the solver's multipliers --
muRows = [muS; 0; nu];            % rows: [cluster1; cluster2 (redundant, 0); next-mode; volume]
K = fp_kkt(P, x, muRows, xsi, eta, 'coneprog reference, solver multipliers');
o.kkt = rmfield(K,'masks');
% and with the implied (stationarity-consistent) box multipliers
K2 = fp_kkt(P, x, muRows, xsiImp, etaImp, 'coneprog reference, implied box multipliers');
o.kkt_implied_box = rmfield(K2,'masks');

% ---- solution description ---------------------------------------------
drho = x(1:NE);
o.solution = struct('bs',x(end),'beta',x(end)*P.lamref,'max_abs_drho',max(abs(drho)), ...
    'max_abs_drho_over_move',max(abs(drho))/P.move,'norm2_drho',norm(drho),'mean_abs_drho',mean(abs(drho)), ...
    'volume_change',sum(drho)/P.Vtot,'volume_row',fval(4),'nextmode_row',fval(3),'cluster1_row',fval(1),'cluster2_row',fval(2), ...
    'image_abc',P.abc(drho).','fJJ_drho',P.fJJ.'*drho,'sum_drho',sum(drho), ...
    'n_positive',nnz(drho>0),'n_negative',nnz(drho<0));
o.active = K.active; o.active_sweep = K.active_sweep; o.active_sweep_cols = K.active_sweep_cols;
o.gain_over_P19 = struct('bs_ref',x(end));

fid = fopen(fullfile(ev,'conic_reference.json'),'w');
fprintf(fid,'%s',jsonencode(o,'PrettyPrint',true)); fclose(fid);
masks = K.masks; muRef = muRows; xsiRef = xsi; etaRef = eta;
save(fullfile(ev,'conic_reference.mat'),'xRef','muRef','xsiRef','etaRef','masks','q','-v7.3');
fprintf('\n[fp_coneprog] exitflag=%d iters=%d wall=%.1fs\n', ef, op.iterations, wall);
fprintf('  bs=%.10f  beta=%.6f  bs-1=%.4e  max|drho|/move=%.4f  ||drho||2=%.4e\n', ...
    o.bs, o.beta, o.bs_minus_1, o.solution.max_abs_drho_over_move, o.solution.norm2_drho);
fprintf('  production fval = %s\n', mat2str(fval.',6));
fprintf('  cone: ||s||=%.6e rhs=%.6e resid=%.3e  separation e2-e1=%.4f (%.3f of d2)\n', ns, rhs, ns-rhs, o.cone.separation_e2_minus_e1, o.cone.separation_over_d2);
fprintf('  duality: primal=%.12f dual=%.12f gap=%.3e  mu=%.6g nu=%s\n', o.duality.primal, o.duality.dual_bound, o.duality.gap, muS, mat2str(nu.',6));
fprintf('  KKT(%s): %s  statRMS=%.3e statMax=%.3e maxfval=%.3e comp=%.3e boxcomp=%.3e\n', 'solver', K.verdict, ...
    K.stationarity.norm_rms, K.stationarity.norm_max, K.primal.max_fval, K.complementarity.max_abs_mu_f, K.complementarity.max_box_comp_normalized);
fprintf('  KKT(implied box): %s statRMS=%.3e\n', K2.verdict, K2.stationarity.norm_rms);
fprintf('  active: -move %d  floor %d  +move %d  ceiling %d  interior %d  (frac move-bound %.4f)\n', ...
    K.active.n_minus_move, K.active.n_floor, K.active.n_plus_move, K.active.n_ceiling, K.active.n_interior, K.active.frac_move_bound);
disp(K.active_sweep);
out = o;
end
