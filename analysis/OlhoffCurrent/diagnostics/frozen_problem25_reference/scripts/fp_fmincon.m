function out = fp_fmincon(which, modes, tag)
%FP_FMINCON  Part 8: independent nonlinear cross-check with fmincon on the
%   EXACT production constraints (deltaLambda + ddlam), interior-point with
%   (A) exact Lagrangian Hessian via HessianMultiplyFcn and (B) L-BFGS, from
%   the preregistered deterministic starts.  FROZEN-SUBPROBLEM REFERENCE --
%   drho is never applied.  which = 'main' (S0,S1,S2,S4,S5,S6) or 'S3'.
if nargin < 1, which = 'main'; end
if nargin < 2 || isempty(modes), modes = {'exact','lbfgs'}; end
if nargin < 3 || isempty(tag), tag = which; end
S = fp_setup();                                   %#ok<NASGU>
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'),'ctx','xP19','xM500'); P = fp_problem(L.ctx);
NE = P.NE; nvar = P.nvar;

% ---- starts --------------------------------------------------------------
bsOf = @(d) (P.lam(1) + P.e1closed(d))/P.lamref;
starts = struct('name',{},'x0',{});
if strcmp(which,'main')
    starts(end+1) = struct('name','S0_zero','x0',[zeros(NE,1);1]);
    starts(end+1) = struct('name','S1_P19','x0',L.xP19);
    starts(end+1) = struct('name','S2_M500','x0',L.xM500);
    d = 0.5*P.xmax(1:NE); starts(end+1) = struct('name','S4_halfmax','x0',[d;bsOf(d)]);
    d = 0.5*P.xmin(1:NE); starts(end+1) = struct('name','S5_halfmin','x0',[d;bsOf(d)]);
    rng(20260912,'twister'); d = P.xmin(1:NE) + (P.xmax(1:NE)-P.xmin(1:NE)).*rand(NE,1);
    starts(end+1) = struct('name','S6_rand','x0',[d;bsOf(d)]);
else
    R = load(fullfile(ev,'mma_replay.mat'),'xM5000');
    starts(end+1) = struct('name','S3_M5000','x0',R.xM5000);
end

% ---- gradient check at S1 (preregistered) --------------------------------
gc = struct();
if strcmp(which,'main')
    rng(20260912,'twister'); xg = L.xP19; h = 1e-6*P.move;
    errs = zeros(5,P.m);
    for k = 1:5
        v = randn(nvar,1); v = v/norm(v);
        [fp_, ~] = P.evalProd(xg + h*v); [fm_, ~] = P.evalProd(xg - h*v);
        [~, dfdx] = P.evalProd(xg);
        fd = (fp_ - fm_)/(2*h); an = dfdx*v;
        errs(k,:) = (abs(fd - an)./max(abs(an),1e-300)).';
    end
    gc = struct('h',h,'max_relerr_per_row',max(errs,[],1),'pass',all(max(errs,[],1) <= 1e-5));
    fprintf('[fp_fmincon] gradient check max relerr per row = %s  pass=%d\n', mat2str(gc.max_relerr_per_row,3), gc.pass);
end

fun = @(x) deal(-x(end), P.f);
nonl = @(x) local_nonl(P, x);
A = P.Alin; b = P.blin;
runs = struct('start',{},'mode',{},'x',{},'fval',{},'exitflag',{},'iterations',{},'funcCount',{}, ...
    'firstorderopt',{},'constrviolation',{},'wall_s',{},'min_sep',{},'message',{},'bs',{},'lambda',{},'kkt',{},'sol',{});
for si = 1:numel(starts)
    for mi = 1:numel(modes)
        mode = modes{mi};
        sepTrack = struct('min',inf,'iters',0);
        outfcn = @(x,ov,state) local_outfcn(P, x, ov, state);
        clear local_outfcn_state
        opts = optimoptions('fmincon','Algorithm','interior-point','Display','final', ...
            'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true, ...
            'OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10,'StepTolerance',1e-14, ...
            'MaxIterations',3000,'MaxFunctionEvaluations',1e5,'OutputFcn',outfcn);
        if strcmp(mode,'exact')
            opts = optimoptions(opts,'HessianMultiplyFcn',@(x,lam,v) P.hessmult(x,lam,v),'SubproblemAlgorithm','cg');
        else
            opts = optimoptions(opts,'HessianApproximation','lbfgs','SubproblemAlgorithm','cg');
        end
        t0 = tic;
        try
            [x, fv, ef, op, lam] = fmincon(fun, starts(si).x0, A, b, [], [], P.xmin, P.xmax, nonl, opts);
            msg = op.message;
        catch ME
            x = nan(nvar,1); fv = NaN; ef = -99; op = struct('iterations',NaN,'funcCount',NaN,'firstorderopt',NaN,'constrviolation',NaN);
            lam = struct('ineqnonlin',nan(2,1),'ineqlin',nan(2,1),'lower',nan(nvar,1),'upper',nan(nvar,1)); msg = ME.message;
        end
        wall = toc(t0);
        st = local_outfcn([],[],[],'get');
        r = struct('start',starts(si).name,'mode',mode,'x',x,'fval',fv,'exitflag',ef, ...
            'iterations',op.iterations,'funcCount',op.funcCount,'firstorderopt',op.firstorderopt, ...
            'constrviolation',op.constrviolation,'wall_s',wall,'min_sep',st.minSep,'message',msg,'bs',x(end));
        r.lambda = struct('ineqnonlin',lam.ineqnonlin(:).','ineqlin',lam.ineqlin(:).', ...
            'lower_max',max(lam.lower),'upper_max',max(lam.upper));
        if all(isfinite(x))
            muRows = [lam.ineqnonlin(:); lam.ineqlin(:)];
            K = fp_kkt(P, x, muRows, lam.lower(:), lam.upper(:), sprintf('fmincon %s %s', starts(si).name, mode));
            r.kkt = rmfield(K,'masks');
            drho = x(1:NE);
            r.sol = struct('bs',x(end),'beta',x(end)*P.lamref,'max_abs_drho_over_move',max(abs(drho))/P.move, ...
                'norm2_drho',norm(drho),'image_abc',P.abc(drho).','fJJ_drho',P.fJJ.'*drho,'sum_drho',sum(drho), ...
                'separation',K.eigs.separation,'active',K.active);
        else
            r.kkt = struct('verdict','NOT_RUN'); r.sol = struct();
        end
        runs(end+1) = r; %#ok<AGROW>
        fprintf('[fp_fmincon] %-10s %-6s ef=%d it=%d bs=%.10f fo=%.2e cv=%.2e wall=%.0fs minsep=%.1f KKT=%s statRMS=%.2e\n', ...
            r.start, mode, ef, op.iterations, x(end), op.firstorderopt, op.constrviolation, wall, st.minSep, r.kkt.verdict, ...
            local_get(r.kkt,'stationarity','norm_rms'));
        X.(sprintf('%s_%s',starts(si).name,mode)) = x;
        R2 = rmfield(runs,'x');
        save(fullfile(ev,sprintf('fmincon_%s.mat',tag)),'X','R2','gc','-v7.3');
        fid = fopen(fullfile(ev,sprintf('fmincon_%s.json',tag)),'w');
        fprintf(fid,'%s',jsonencode(struct('gradient_check',gc,'runs',R2),'PrettyPrint',true)); fclose(fid);
    end
end
out = struct('gradient_check',gc,'runs',rmfield(runs,'x'));
end

function v = local_get(s, a, b)
if isfield(s,a) && isfield(s.(a),b), v = s.(a).(b); else, v = NaN; end
end
function [c, ceq, gc, gceq] = local_nonl(P, x)
[fval, dfdx] = P.evalProd(x);
c = fval(1:2); ceq = [];
gc = dfdx(1:2,:).'; gceq = [];
end
function stop = local_outfcn(P, x, ov, state)
persistent st
if ischar(state) && strcmp(state,'get'), stop = st; return, end
if strcmp(state,'init') || isempty(st), st = struct('minSep',inf,'iters',0); end
stop = false;
if ~isempty(x)
    [~,~,r] = P.uvr(x(1:P.NE)); sep = 2*r;
    st.minSep = min(st.minSep, sep); st.iters = ov.iteration;
end
end
