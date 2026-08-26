function res = olhoffOpt(cfg)
%OLHOFFOPT  Du & Olhoff (2007) sec. 3.5 -- maximization of the n-th
%   eigenfrequency, problem (25).  Main loop of Fig. 1.
%
%   Every unstated quantity is taken from cfg and echoed into res.cfg so that
%   each figure is traceable to a config (CLAUDE.md sec.4 and sec.8).

maxNumCompThreads(cfg.threads);
t0 = tic;

mdl = model2D(cfg);
NE  = mdl.nele;
rho = cfg.rho0*ones(NE,1);

% Filter radius. The paper never states it. It is specified physically where
% possible so that it is MESH-INDEPENDENT: rminEl must scale with the element
% size or a mesh refinement silently changes the filter as well.
if isfield(cfg,'rminPhys') && ~isempty(cfg.rminPhys) && cfg.rminPhys > 0
    dyEl = cfg.b/cfg.nely;
    cfg.rminEl = cfg.rminPhys/dyEl;
end
flt = prepFilter(cfg.nelx, cfg.nely, cfg.rminEl);

n     = cfg.n;
Nmax  = cfg.Nmax;
Jcalc = n + Nmax;

hist = struct('omega',[],'N',[],'beta',[],'nInner',[],'dxOuter',[], ...
              'vol',[],'tEig',[],'tGrad',[],'tInner',[],'degen',[],'multJ',[], ...
              'innerConv',[],'cumInner',[]);
log = {};
cumInner = 0;

if cfg.verbose
    fprintf('%4s %9s %9s %9s %4s %9s %6s %6s %8s %9s %7s\n', ...
            'it','omega1','omega2','omega3','N','sqrt(beta)', ...
            'inner','cumIn','maxdrho','vol','conv');
end

for outer = 1:cfg.maxOuter
    % ---- step 1: FE analysis + multiplicity detection -------------------
    te = tic;
    [K,M] = assemble2D(mdl, rho, cfg.p, cfg.massInterp);
    [w, Phi, lam] = eigSolve(K, M, Jcalc, cfg.solver);
    tEig = toc(te);

    N = 1;
    while n+N <= Jcalc-1 && abs(w(n+N)-w(n))/w(n) < cfg.tolMult
        N = N + 1;
    end
    if N >= Nmax
        log{end+1} = sprintf('iter %d: detected N=%d >= Nmax=%d, J may be truncated',outer,N,Nmax); %#ok<AGROW>
    end
    J = n + N;
    multJ = (J+1 <= Jcalc) && abs(w(J+1)-w(J))/w(J) < cfg.tolMult;
    if multJ
        % (25b) assumes omega_J simple.  No procedure defined -- log only.
        log{end+1} = sprintf('iter %d: omega_J (J=%d) is itself multiple -- (25b) undefined',outer,J); %#ok<AGROW>
    end

    % ---- step 2: generalized gradients ----------------------------------
    tg = tic;
    idx     = n:(n+N-1);
    lamTild = mean(lam(idx));
    F       = genGrad(mdl, rho, cfg.p, cfg.massInterp, Phi, lamTild, idx);
    FJ      = genGrad(mdl, rho, cfg.p, cfg.massInterp, Phi, lam(J), J);
    fJJ     = FJ(:,1,1);

    % ---- filtering (Sigmund 1997, applied to the sensitivities) ---------
    switch lower(cfg.filterMode)
        case 'diag'      % filter only the f_jj (and f_JJ)
            for j = 1:N
                F(:,j,j) = applyFilter(flt, rho, F(:,j,j));
            end
        case 'all'       % filter every f_sk, including off-diagonals
            for s = 1:N
                for k = s:N
                    v = applyFilter(flt, rho, F(:,s,k));
                    F(:,s,k) = v;  F(:,k,s) = v;
                end
            end
        case 'none'
        otherwise
            error('olhoffOpt:filterMode','unknown filterMode %s',cfg.filterMode);
    end
    fJJ = applyFilter(flt, rho, fJJ);
    tGrad = toc(tg);

    % ---- step 3: inner loop ---------------------------------------------
    ti = tic;
    ctx = struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J), ...
                 'rho',rho,'rhomin',cfg.rhomin,'volfrac',cfg.volfrac, ...
                 'move',cfg.move,'maxInner',cfg.maxInner, ...
                 'tolInner',cfg.tolInner,'minInner',cfg.minInner, ...
                 'offDiag',cfg.offDiag);
    if strcmpi(cfg.innerSolver,'lp')
        [drho, st] = innerLoopLP(ctx);
        if ~st.conv
            log{end+1} = sprintf('iter %d: LP inner solve failed (flag=%d)',outer,st.lpFlag); %#ok<AGROW>
        end
    else
        [drho, st] = innerLoop(ctx);
    end
    tInner = toc(ti);

    % ---- step 4: update --------------------------------------------------
    rho = min(1, max(cfg.rhomin, rho + drho));
    dxOuter = max(abs(drho));

    hist.omega(:,outer)  = w(1:min(Jcalc,numel(w)));
    hist.N(outer)        = N;
    hist.beta(outer)     = st.beta;
    hist.nInner(outer)   = st.nInner;
    cumInner             = cumInner + st.nInner;
    hist.cumInner(outer) = cumInner;
    hist.innerConv(outer)= st.conv;
    hist.dxOuter(outer)  = dxOuter;
    hist.vol(outer)      = mean(rho);
    hist.tEig(outer)     = tEig;
    hist.tGrad(outer)    = tGrad;
    hist.tInner(outer)   = tInner;
    hist.degen(outer)    = st.degenHits;
    hist.multJ(outer)    = multJ;

    if cfg.verbose
        fprintf('%4d %9.2f %9.2f %9.2f %4d %9.2f %6d %6d %8.4f %9.3f %7s\n', ...
                outer, w(1), w(2), w(min(3,end)), N, sqrt(max(st.beta,0)), ...
                st.nInner, cumInner, dxOuter, mean(rho), ...
                stringYesNo(st.conv));
    end

    if dxOuter < cfg.tolOuter
        log{end+1} = sprintf('converged at outer iteration %d (max|drho| = %.3e)',outer,dxOuter); %#ok<AGROW>
        break
    end
end

% ---- final analysis -----------------------------------------------------
[K,M] = assemble2D(mdl, rho, cfg.p, cfg.massInterp);
[w, Phi, lam] = eigSolve(K, M, Jcalc, cfg.solver);
T = classifyModes(mdl, M, Phi, w);

res = struct('cfg',cfg,'rho',rho,'omega',w,'lambda',lam,'hist',hist, ...
             'modeTable',T,'log',{log},'nOuter',numel(hist.N), ...
             'wallclock',toc(t0),'mdl',mdl);
end

function s = stringYesNo(tf)
if tf, s = 'yes'; else, s = 'NO'; end
end
