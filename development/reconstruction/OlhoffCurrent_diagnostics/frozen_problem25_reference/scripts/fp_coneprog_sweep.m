function out = fp_coneprog_sweep()
%FP_CONEPROG_SWEEP  Part 7 continued: the preregistered 1e-10 coneprog call
%   stalled (exitflag -7, duals unconverged).  Sweep the linear-solver choice
%   and an EXACT affine unit-box reparametrization  x = xmin + W t, t in
%   [0,1], to obtain a strongly terminated solve.  Every candidate is scored
%   by the independent maximized dual bound of fp_dualbound, never by the
%   solver's own status alone.  Disclosed as a deviation from the single
%   preregistered call.  drho is never applied.
S = fp_setup();                                   %#ok<NASGU>
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'),'ctx'); P = fp_problem(L.ctx);
NE = P.NE; nvar = P.nvar;

W = P.xmax - P.xmin; Wd = spdiags(W,0,nvar,nvar);
% t-form:  x = xmin + W t
Ac_t = P.Ac*Wd; bc_t = P.bc - P.Ac*P.xmin; dc_t = Wd*P.dc; g_t = P.gammac - P.dc.'*P.xmin;
Alin_t = P.Alin*Wd; blin_t = P.blin - P.Alin*P.xmin; f_t = Wd*P.f;
% pointwise check of the t-form against the x-form at the test points
T = fp_testpoints(P, ev); chk = zeros(numel(T),3);
for i = 1:numel(T)
    x = [T(i).drho; 1.001]; t = (x - P.xmin)./W;
    chk(i,:) = [abs((norm(Ac_t*t-bc_t)-(dc_t.'*t-g_t)) - P.coneResid(x)), ...
                max(abs((Alin_t*t-blin_t)-(P.Alin*x-P.blin))), abs(f_t.'*t + P.f.'*P.xmin - P.f.'*x)];
end
o = struct(); o.tform_check_max = max(chk,[],1);

solvers = {'auto','prodchol','schur','augmented','normal'};
forms = {'x','t'};
tols = [1e-10 1e-8];
cands = struct('form',{},'solver',{},'tol',{},'exitflag',{},'iterations',{},'wall_s',{},'bs',{}, ...
    'max_fval',{},'box_viol',{},'dual_bound',{},'gap',{},'mu',{},'nu',{},'message',{});
X = struct();
for fi = 1:numel(forms)
    for si = 1:numel(solvers)
        for ti = 1:numel(tols)
            opts = optimoptions('coneprog','Display','off','OptimalityTolerance',tols(ti), ...
                'ConstraintTolerance',tols(ti),'MaxIterations',500,'LinearSolver',solvers{si});
            t0 = tic;
            try
                if strcmp(forms{fi},'x')
                    soc = secondordercone(P.Ac, P.bc, P.dc, P.gammac);
                    [x, ~, ef, op, lam] = coneprog(P.f, soc, P.Alin, P.blin, [], [], P.xmin, P.xmax, opts);
                else
                    soc = secondordercone(Ac_t, bc_t, dc_t, g_t);
                    [t, ~, ef, op, lam] = coneprog(f_t, soc, Alin_t, blin_t, [], [], zeros(nvar,1), ones(nvar,1), opts);
                    x = P.xmin + W.*t;
                end
                msg = op.message;
            catch ME
                x = nan(nvar,1); ef = -99; op = struct('iterations',NaN); lam = struct('soc',NaN,'ineqlin',[NaN;NaN]); msg = ME.message;
            end
            wall = toc(t0);
            c = struct('form',forms{fi},'solver',solvers{si},'tol',tols(ti),'exitflag',ef,'iterations',op.iterations,'wall_s',wall);
            if all(isfinite(x))
                x = min(P.xmax, max(P.xmin, x));   % clip roundoff-level bound violations before scoring
                fval = P.evalProd(x);
                mu = lam.soc; if iscell(mu), mu = mu{1}; end
                D = fp_dualbound(P, x, max(mu(1),0), lam.ineqlin);
                c.bs = x(end); c.max_fval = max(fval); c.box_viol = max([max(P.xmin-x), max(x-P.xmax), 0]);
                c.dual_bound = D.dual_bound; c.gap = D.gap; c.mu = D.mu; c.nu = D.nu;
                X.(sprintf('%s_%s_%g',forms{fi},solvers{si},-log10(tols(ti)))) = x;
            else
                c.bs = NaN; c.max_fval = NaN; c.box_viol = NaN; c.dual_bound = NaN; c.gap = NaN; c.mu = NaN; c.nu = [NaN NaN];
            end
            c.message = msg;
            cands(end+1) = c; %#ok<AGROW>
            fprintf('[sweep] form=%s solver=%-9s tol=%g ef=%3d it=%3d bs=%.12f maxfval=%.2e gap=%.3e wall=%.1fs\n', ...
                c.form, c.solver, c.tol, ef, op.iterations, c.bs, c.max_fval, c.gap, wall);
        end
    end
end
o.candidates = cands;
% selection: feasible (max_fval <= 1e-8), prefer exitflag 1, then smallest gap
sc = arrayfun(@(c) (c.max_fval <= 1e-8)*1e6 + (c.exitflag == 1)*1e3 - min(c.gap,1e3)*1e3, cands);
sc(~isfinite(sc)) = -inf;
[~, ib] = max(sc); o.selected = cands(ib); o.selected_index = ib;
fid = fopen(fullfile(ev,'coneprog_sweep.json'),'w');
fprintf(fid,'%s',jsonencode(o,'PrettyPrint',true)); fclose(fid);
save(fullfile(ev,'coneprog_sweep.mat'),'X','cands','-v7.3');
fprintf('[sweep] selected: form=%s solver=%s tol=%g ef=%d bs=%.12f gap=%.3e\n', o.selected.form, o.selected.solver, o.selected.tol, o.selected.exitflag, o.selected.bs, o.selected.gap);
out = o;
end
