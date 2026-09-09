function res = olhoffSolve(cfg)
%OLHOFFSOLVE  Du & Olhoff (2007) sec. 3.5 -- maximization of the n-th
%   eigenfrequency, problem (25).  Main loop of Fig. 1.
%
%   res = OLHOFFSOLVE(cfg) with cfg a CANONICAL configuration, as produced by
%   olh.config.resolve.  The configuration is immutable here: this function
%   never writes to it and never supplies a default.  Every question it asks of
%   the configuration is a scientific question, phrased in scientific terms.
%
%   There is no experiment identifier anywhere below.  What used to be
%       cfg.restorationGuard == 'R2'          is  cfg.stop.guards.maxDesignChange
%       cfg.multRule == 'subspace'  (for the
%           diagonal-offset form of (25d))    is  cfg.multiplicity.diagonalOffsets
%       cfg.projection.on -> density filter   is  cfg.filter.type == 'density'
%       cfg.moveFamily == 'S2'                is  cfg.move.policy == 'ladder'
%
%   ARITHMETIC IS UNCHANGED.  Every numeric expression is character-identical to
%   algo/olhoffOpt.m at baseline 2029baa.  Only configuration reads and branch
%   predicates differ, so the trajectory is bitwise reproducible.
%
%   See also OLH.CONFIG.RESOLVE, OLHOFFOPT (the legacy-flat compatibility shim).

g = @(p) olh.config.getPath(cfg, p);

if g('runtime.singleThread'), maxNumCompThreads(1); end
t0 = tic;

useMMA(g('optimizer.inner.variant'));   % selects which mmasub resolves

% The FE model and the numerical kernels still read the flat field names; that
% is deliberate, because leaving their arithmetic untouched is what makes the
% trajectory reproducible.  One rendering, built once, never mutated.
flat = olh.config.toLegacy(cfg);

mdl = model2D(flat);
NE  = mdl.nele;
rho = g('design.initial')*ones(NE,1);

% ---- filter -------------------------------------------------------------
% The radius is specified PHYSICALLY where possible so that it is
% MESH-INDEPENDENT: an element-count radius would silently change the filter
% under refinement.  The paper never states a radius, for any example.
rminPhys = g('filter.radiusPhysical');
if ~isempty(rminPhys) && rminPhys > 0
    dyEl   = g('domain.b')/g('domain.mesh.nely');
    rminEl = rminPhys/dyEl;
else
    rminEl = g('filter.radiusElements');
end
flt = prepFilter(g('domain.mesh.nelx'), g('domain.mesh.nely'), rminEl);

n     = g('eigen.targetMode');
Nmax  = g('eigen.maxCluster');
Jcalc = n + Nmax;

rhomin  = g('design.minimum');
volfrac = g('design.volumeFraction');
pFixed  = g('material.stiffness.p');
massTerminal = g('material.mass.model');

hist = struct('omega',[],'N',[],'beta',[],'nInner',[],'dxOuter',[], ...
              'vol',[],'tEig',[],'tGrad',[],'tInner',[],'degen',[],'multJ',[], ...
              'innerConv',[],'cumInner',[],'dxNorm2',[],'move',[],'gap12',[], ...
              'volErr',[],'dBeta',[],'stage',[],'pPen',[],'pStage',[],'pEvent',[],'massLow',[], ...
              'projBeta',[],'projStage',[],'projEvent',[],'dxPhys2',[], ...
              ... % stage-exhaustion controller trace.  Written only when the
              ... % controller is selected; all NaN/false otherwise.  Read back
              ... % by nothing inside the solve.
              'exA',[],'exB',[],'exE',[],'exNA',[],'exNB',[],'exDecl',[], ...
              'exCos',[],'exNet',[],'exMedcos',[],'exMednet',[],'exAmp',[], ...
              'exStageStart',[], ...
              ... % BENCHMARK TIMING INSTRUMENTATION ONLY.  Carried forward from
              ... % the frozen conference reconstruction, where it is recorded as
              ... % patches/olhoffOpt.timing-instrumentation.diff.  tOuter is the
              ... % wall time of one complete outer iteration, inner solve
              ... % included; it is WRITTEN AND NEVER READ BACK, so the trajectory
              ... % is unchanged.  See analysis/OlhoffCurrent/PROVENANCE.md sec.7.
              'tOuter',[]);
log = {};
cumInner = 0;
wantDiag = g('runtime.diagnostics');
dg = struct('drho',{{}},'dlamPred',{{}},'lam',{{}},'N',[],'gap',[], ...
            'fdiag',[],'foff',[],'Vrot',[],'lamTild',[],'beta',[]);
mmaState = [];   % persistent MMA solver state (inner variable 'design' only)
multState = [];  % multiplicity-rule state
mvState  = [];   % move controller state
lamPrev  = [];   % previous lambda_n, for the trust-ratio realized-gain measure

% ---- policies in force, decided ONCE -------------------------------------
pContinuation = g('material.stiffness.continuation.enabled');
pSchedule     = g('material.stiffness.continuation.schedule');
pOwnCounter   = pContinuation && strcmp(g('material.stiffness.continuation.driver'),'ownCounter');
pBlocksStop   = pContinuation && g('material.stiffness.continuation.blockStopUntilFinal');
massContinuation = g('material.mass.continuation.enabled');
massLowModel     = g('material.mass.continuation.lowPModel');
useProj       = g('projection.enabled');
projLevels    = g('projection.beta.levels');
projEta       = g('projection.eta');
useDensityFilter = strcmp(g('filter.type'),'density');
useSensFilter    = strcmp(g('filter.type'),'sensitivity');
filterAll        = strcmp(g('filter.applyTo'),'all');
useOff        = g('multiplicity.diagonalOffsets');
offDiag       = g('multiplicity.offDiagonal');
moveLevels    = g('move.levels');
guardSettled  = g('stop.guards.settledMove');
guardLadder   = g('stop.guards.ladderExhausted');
guardMaxChange= g('stop.guards.maxDesignChange');
anyStopGuard  = guardLadder || guardMaxChange;
tolOuter      = g('stop.tolerance');
stopNormL2    = strcmp(g('stop.norm'),'l2');
% ---- the stage-exhaustion controller, if selected ------------------------
% Two INDEPENDENT switches, both defaulting to the historical behaviour, so a
% configuration that names neither is bitwise the solver that existed before:
%   move.continuation.signal == 'stageExhaustion'  the ladder descends on the
%       frozen two-branch rule instead of on the bound variable's stall;
%   stop.rule == 'stageExhaustion'                 outer convergence is admitted
%       only at the last move level, and only once that same frozen rule has
%       been satisfied with its full persistence.
% Both read olh.move.exhaustion, which is advanced once per outer iteration
% immediately after the design update.
exhaustMove   = strcmp(g('move.continuation.signal'),'stageExhaustion');
exhaustStop   = strcmp(g('stop.rule'),'stageExhaustion');
useExhaustion = exhaustMove || exhaustStop;
maxOuter      = g('runtime.maxOuter');
verbose       = g('runtime.verbose');
innerLP       = strcmp(g('optimizer.inner.type'),'lp');
innerDesignVar= strcmp(g('optimizer.inner.variable'),'design');
massCfg       = g('material.mass');

if verbose
    fprintf('%4s %9s %9s %9s %4s %9s %6s %6s %8s %9s %8s %9s %7s\n', ...
            'it','omega1','omega2','omega3','N','sqrt(beta)', ...
            'inner','cumIn','maxdrho','|drho|2','move','vol','conv');
end

% The detector's quantities are formed from the design variable and from the
% increment the sub-problem returned.  Under projection those are z and dz, not
% the physical density, and the frozen rule was never defined there.  Refuse
% rather than silently measure a different thing.
if useExhaustion && useProj
    error('olh:stop:exhaustionUnderProjection', ...
        ['the two-branch stage-exhaustion rule is defined on the design ' ...
         'variable of the unprojected formulation; projection.enabled must be ' ...
         'false when move.continuation.signal or stop.rule is ''stageExhaustion''.']);
end

pStage    = 1;    % index into the p schedule
prevStage = 1;    % move-ladder stage seen at the previous iteration
projStage = 1;    % index into the projection sharpness levels

% ---- discreteness projection --------------------------------------------
% When enabled the design variable becomes z and the physical density is the
% three-field map  z -> zTilde = (H z)/Hs -> rhoPhys = rhomin+(1-rhomin)*P(.),
% with P the tanh projection.  rhoPhys is used by the stiffness assembly, the
% mass interpolation, the volume constraint and the eigenvalue evaluation, and
% ALL eigenvalue sensitivities -- diagonal AND off-diagonal -- are carried to z
% by the complete chain rule.
%
% CLASS D.  Absent from Du & Olhoff (2007), the erratum, Olhoff & Du (2014) and
% the Krog & Olhoff lineage.
z = [];  sChain = [];
if useProj
    z = rho;                      % design variable, box [0,1]
end

for outer = 1:maxOuter
    tOuterTic = tic;   % BENCHMARK TIMING INSTRUMENTATION ONLY

    % ---- SIMP penalization in force this iteration ----------------------
    % Sec. 2.1: p is "normally assigned values increasing from 1 to 3 during the
    % optimization process".  The paper gives no schedule and no transition
    % rule, so the transition reuses the stall event the move controller already
    % computes -- no new numerical constant is introduced.
    %
    % driver 'moveLadderStage': p is indexed by the ladder stage itself.
    % driver 'ownCounter':      p advances on its own counter, consuming the
    %                           SAME stall event but not sharing the index.
    pNow = pFixed;
    if pContinuation
        if pOwnCounter
            pNow = pSchedule(min(pStage, numel(pSchedule)));
        else
            if isempty(mvState) || ~isfield(mvState,'stage'); stageInForce = 1;
            else;                                            stageInForce = mvState.stage; end
            pNow = pSchedule(min(stageInForce, numel(pSchedule)));
        end
    end
    hist.pPen(outer)   = pNow;
    hist.pStage(outer) = pStage;

    % ---- mass model in force this iteration -----------------------------
    % The low-p model is in force exactly while p is below its final value, and
    % the terminal model resumes when p first reaches it.  The switch introduces
    % no numerical threshold of its own: it is tied to the existing p schedule.
    massNow = massTerminal;
    if massContinuation && pNow < pSchedule(end)
        massNow = massLowModel;
    end
    hist.massLow(outer) = ~strcmp(massNow, massTerminal);
    massNowCfg = massCfg;  massNowCfg.model = massNow;

    % ---- projection sharpness in force this iteration -------------------
    bProj = 0;
    if useProj
        bProj = projLevels(min(projStage, numel(projLevels)));
        [rho, sChain] = projDensityField(flt.H, flt.Hs, z, bProj, projEta, rhomin);
    end
    hist.projBeta(outer)  = bProj;
    hist.projStage(outer) = projStage;

    % ---- step 1: FE analysis + multiplicity detection -------------------
    te = tic;
    [K,M] = assemble2D(mdl, rho, pNow, massNowCfg);
    [w, Phi, lam] = eigSolve(K, M, Jcalc, g('eigen.solver'));
    tEig = toc(te);

    [N, multState] = olh.multi.detect(cfg, w, n, Jcalc, multState);
    if N >= Nmax
        log{end+1} = sprintf('iter %d: detected N=%d >= Nmax=%d, J may be truncated',outer,N,Nmax); %#ok<AGROW>
    end
    J = n + N;
    multJ = (J+1 <= Jcalc) && abs(w(J+1)-w(J))/w(J) < g('multiplicity.tolerance');
    if multJ
        % (25b) assumes omega_J simple.  No procedure defined -- log only.
        log{end+1} = sprintf('iter %d: omega_J (J=%d) is itself multiple -- (25b) undefined',outer,J); %#ok<AGROW>
    end

    % ---- step 2: generalized gradients ----------------------------------
    tg = tic;
    idx     = n:(n+N-1);
    % Sec. 3.5.1: "In the second step of the main loop, we set lambda~ = omega_n^2"
    % -- the FIRST eigenvalue of the cluster, NOT the cluster mean.
    lamTild = lam(n);
    F       = genGrad(mdl, rho, pNow, massNowCfg, Phi, lamTild, idx);
    % Diagonal-offset form: the subeigenvalue problem keeps the actual
    % separation lam(j)-lam(n), so the diagonal blocks use each mode's OWN
    % eigenvalue -- eq. (24) verbatim, f_jj with lambda_j.  The off-diagonals
    % keep (19)'s lambda-tilde.  At exact degeneracy the two coincide and the
    % whole thing collapses onto (25d) as printed.
    if useOff
        for j = 1:N
            Gj = genGrad(mdl, rho, pNow, massNowCfg, Phi, lam(idx(j)), idx(j));
            F(:,j,j) = Gj(:,1,1);
        end
        dOff = lam(idx) - lam(idx(1));
    else
        dOff = [];
    end
    FJ      = genGrad(mdl, rho, pNow, massNowCfg, Phi, lam(J), J);
    fJJ     = FJ(:,1,1);

    % ---- sensitivity treatment ------------------------------------------
    if useDensityFilter
        % Complete chain rule  d/dz = W' S d/drho, applied to EVERY generalized
        % gradient -- diagonal AND off-diagonal -- and to f_JJ.  a_sk is a scalar
        % function of rho, so the chain rule does not distinguish s==k from s~=k;
        % filter.applyTo is therefore not consulted on this path.  The transform
        % precedes deltaLambda so that the sub-eigenvalue matrix A, its
        % eigenvectors and its gradients are all expressed in z.
        %
        % This REPLACES the Sigmund (1997) sensitivity filter.  Sec. 1 states the
        % filter was applied to the SENSITIVITIES, so this is a departure from a
        % printed choice, and cfg.filter.type is where that departure is declared.
        for s = 1:N
            for k = s:N
                v = projChain(flt.H, flt.Hs, sChain, F(:,s,k));
                F(:,s,k) = v;  F(:,k,s) = v;
            end
        end
        fJJ  = projChain(flt.H, flt.Hs, sChain, fJJ);
    elseif useSensFilter
    % ---- filtering (Sigmund 1997, applied to the sensitivities) ---------
    if filterAll        % every f_sk, including off-diagonals
            for s = 1:N
                for k = s:N
                    v = applyFilter(flt, rho, F(:,s,k));
                    F(:,s,k) = v;  F(:,k,s) = v;
                end
            end
    else                % only the f_jj (and f_JJ)
            for j = 1:N
                F(:,j,j) = applyFilter(flt, rho, F(:,j,j));
            end
    end
    fJJ = applyFilter(flt, rho, fJJ);
    end
    tGrad = toc(tg);

    % ---- step/move control ----------------------------------------------
    % NONE of this is specified by the paper: the only printed bound on drho is
    % the box (25f).
    if ~isempty(lamPrev)
        mvState.lastRealized = lam(n) - lamPrev;
    elseif ~isempty(mvState)
        mvState.lastRealized = NaN;
    end
    [mvNow, mvState] = olh.move.limit(cfg, outer, hist, mvState);
    % ---- p continuation on its own counter: intercept the stall event ---
    % A stall is detected here as a ladder stage increase.  While p has not
    % reached its final value that event is consumed by the p controller INSTEAD
    % of the ladder: p advances one step, the move is restored to the ladder's
    % own first level, and the ladder index and its re-arm clock are reset so the
    % existing dwell rule applies.  Once p is final the event is left to the
    % ladder and refinement proceeds normally.
    pEvent = 0;
    if pOwnCounter
        stageNow = 1; if isfield(mvState,'stage'), stageNow = mvState.stage; end
        if stageNow > prevStage && pStage < numel(pSchedule)
            pStage        = pStage + 1;
            mvState.stage = 1;
            mvState.lastStage = outer;
            mvNow         = moveLevels(1);
            pEvent        = 1;
            log{end+1} = sprintf(['iter %d: stall consumed by p continuation -> p=%.4g ' ...
                'at next iteration; move restored to %.4g, S2 stage reset to 1'], ...
                outer, pSchedule(pStage), mvNow); %#ok<AGROW>
        end
        stageNow = 1; if isfield(mvState,'stage'), stageNow = mvState.stage; end
        prevStage = stageNow;
    end
    hist.pEvent(outer) = pEvent;
    lamPrev = lam(n);

    % ---- step 3: inner loop ---------------------------------------------
    ti = tic;
    ctx = struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J), ...
                 'rho',rho,'rhomin',rhomin,'volfrac',volfrac, ...
                 'move',mvNow,'maxInner',g('optimizer.inner.maxIterations'), ...
                 'tolInner',g('optimizer.inner.tolerance'), ...
                 'minInner',g('optimizer.inner.minIterations'), ...
                 'offDiag',offDiag,'dOff',dOff);
    if useProj
        % The optimization variable is z, so the MMA box and the move limit
        % bound dz:  max(0-z,-move) <= dz <= min(1-z,+move).  Passing z through
        % ctx.rho and 0 through ctx.rhomin reuses the box code verbatim.  The
        % volume constraint (25e) is evaluated exactly at z+dz through volFun.
        ctx.rho    = z;
        ctx.rhomin = 0;
        ctx.volFun = @(dz) local_projVolume(flt.H, flt.Hs, z, dz, bProj, projEta, rhomin);
    end
    if innerLP
        [drho, st] = innerLoopLP(ctx);
        if ~st.conv
            log{end+1} = sprintf('iter %d: LP inner solve failed (flag=%d)',outer,st.lpFlag); %#ok<AGROW>
        end
    elseif innerDesignVar
        [drho, st, mmaState] = innerLoopRho(ctx, mmaState);
    else
        [drho, st] = innerLoop(ctx);
    end
    tInner = toc(ti);

    % ---- step 4: update --------------------------------------------------
    % Under projection the update is applied to the DESIGN VARIABLE z (drho
    % holds dz), and the physical density is recomputed from the map so that
    % hist.vol records the post-update PHYSICAL volume.
    if useProj
        z   = min(1, max(0, z + drho));
        rhoPrev = rho;
        [rho, sChain] = projDensityField(flt.H, flt.Hs, z, bProj, projEta, rhomin);
        hist.dxPhys2(outer) = norm(rho - rhoPrev);   % cross-reference only
    else
        rho = min(1, max(rhomin, rho + drho));
        hist.dxPhys2(outer) = NaN;
    end
    % cfg.stop.field is 'designVariable': BOTH measures below are formed from
    % drho, which is d(design variable) -- d(rho) without projection and d(z)
    % with it.  Sec. 3.5.1 monitors the design increment, so this is faithful,
    % but under projection it is NOT the change in physical density; that is
    % hist.dxPhys2.
    dxOuter = max(abs(drho));
    dxNorm2 = norm(drho);

    if wantDiag
        dg.drho{end+1}  = drho;
        dg.lam{end+1}   = lam(1:min(Jcalc,numel(lam)));
        dg.N(end+1)     = N;
        dg.gap(end+1)   = (w(2)-w(1))/w(1);
        dg.lamTild(end+1)= lamTild;
        dg.beta(end+1)  = st.beta;
        % predicted increments along the realized step, and the scale of the
        % (25d) matrix entries.  Pure post-hoc evaluation: nothing feeds back.
        dg.dlamPred{end+1} = deltaLambda(F, drho, dOff);
        A = zeros(N);
        for s_ = 1:N, for k_ = 1:N, A(s_,k_) = F(:,s_,k_).'*drho; end, end
        dg.fdiag(end+1) = max(abs(diag(A)));
        if N>=2, dg.foff(end+1) = max(abs(A(~eye(N))));
        else,    dg.foff(end+1) = 0; end
        if outer>1 && ~isempty(dg.drho{end-1})
            u = dg.drho{end-1}; v = drho;
            dg.Vrot(end+1) = (u.'*v)/max(norm(u)*norm(v),1e-300);
        else
            dg.Vrot(end+1) = NaN;
        end
    end
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
    hist.dxNorm2(outer)  = dxNorm2;
    hist.move(outer)     = mvNow;
    hist.gap12(outer)    = (w(2)-w(1))/w(1);
    hist.volErr(outer)   = mean(rho) - volfrac;
    if outer > 1, hist.dBeta(outer) = st.beta - hist.beta(outer-1);
    else,         hist.dBeta(outer) = NaN; end
    if isfield(mvState,'stage'), hist.stage(outer) = mvState.stage;
    else,                        hist.stage(outer) = 1; end

    if verbose
        fprintf('%4d %9.2f %9.2f %9.2f %4d %9.2f %6d %6d %8.4f %9.4f %8.4f %9.3f %7s\n', ...
                outer, w(1), w(2), w(min(3,end)), N, sqrt(max(st.beta,0)), ...
                st.nInner, cumInner, dxOuter, dxNorm2, mvNow, mean(rho), ...
                local_yesno(st.conv));
    end

    % ---- stage-exhaustion detector ---------------------------------------
    % Advanced ONCE per outer iteration, here, so that the convergence test
    % below sees information through this iteration and olh.move.limit sees
    % information through the previous one -- the same causal structure the
    % bound-variable stall detector has.  It is a pure observer of rho and drho.
    if useExhaustion
        mvState.ex = olh.move.exhaustion(mvState.ex, NE, tolOuter, outer, ...
                                         rho, drho, g('design.initial'));
        ex = mvState.ex;
        hist.exA(outer)      = ex.A(outer);
        hist.exB(outer)      = ex.B(outer);
        hist.exE(outer)      = ex.E(outer);
        hist.exNA(outer)     = ex.nA(outer);
        hist.exNB(outer)     = ex.nB(outer);
        hist.exDecl(outer)   = double(ex.declared);
        hist.exCos(outer)    = ex.cos(outer);
        hist.exNet(outer)    = ex.net(outer);
        hist.exMedcos(outer) = ex.medcos(outer);
        hist.exMednet(outer) = ex.mednet(outer);
        hist.exAmp(outer)    = ex.amp(outer);
        hist.exStageStart(outer) = ex.stageStart;
    end

    % ---- convergence metric ---------------------------------------------
    % Sec. 3.5.1 tests "the norm of the vector drho ... less than a small,
    % predefined value epsilon".  The norm is unqualified; l2 is the natural
    % reading and 'max' is kept as a labelled alternative.  epsilon itself is
    % unstated and is a per-run recorded parameter.
    if stopNormL2, convOuter = dxNorm2 < tolOuter;
    else,          convOuter = dxOuter < tolOuter;
    end

    % ---- guard: the metric may only be believed on a settled move -------
    % The paper places no bound on drho other than the box (25f), so there
    % ||drho|| -> 0 genuinely means the design has stopped moving.  This
    % reconstruction adds a move limit, and then ||drho||_inf <= mv_k: on an
    % iteration where mv_k differs from mv_{k-1} a scheduled reduction of the
    % move limit mechanically reduces the measured step with no change in the
    % design's behaviour, and the criterion is uninterpretable there.
    if guardSettled && ~exhaustStop
        moveSettled = outer >= 2 && hist.move(outer) == hist.move(outer-1);
        if convOuter && ~moveSettled
            log{end+1} = sprintf(['iter %d: ||drho|| below eps but the move limit ' ...
                'just changed (%.6g -> %.6g); convergence NOT asserted'], ...
                outer, local_prevMove(hist,outer), hist.move(outer)); %#ok<AGROW>
        end
        convOuter = convOuter && moveSettled;
    end

    % ---- guards: additional conditions on admitting convergence ---------
    % Neither guard changes the update or the move controller.
    if anyStopGuard && ~exhaustStop
        epsRMS = tolOuter/sqrt(NE);
        restorationReady = true;
        if guardLadder
            % Is any level the ladder has NOT yet reached still large enough to
            % produce a change the tolerance would call significant?  A statement
            % about the SCHEDULE.
            restorationReady = restorationReady && ...
                ~any(moveLevels(hist.stage(outer)+1:end) > epsRMS);
        end
        if guardMaxChange
            % Has the largest single design change fallen below the same RMS
            % scale the outer tolerance uses?  A statement about the DESIGN.
            restorationReady = restorationReady && (dxOuter < epsRMS);
        end
        if convOuter && ~restorationReady
            log{end+1} = sprintf('iter %d: baseline stop blocked by %s (stage=%d, max|drho|=%.6g, epsRMS=%.6g)', ...
                outer,local_guardName(guardLadder,guardMaxChange), ...
                hist.stage(outer),dxOuter,epsRMS); %#ok<AGROW>
        end
        convOuter = convOuter && restorationReady;
    end

    % ---- stage-exhaustion terminal admission -----------------------------
    % At the last move level there is no lower rung, so the SAME scientific
    % concept that descends the ladder must govern terminal admission: the run
    % may stop only once the terminal stage has itself satisfied the frozen
    % rule with its full persistence.  The sec. 3.5.1 design-change test and its
    % settled-move / restoration guards are production's rule and are replaced
    % wholesale, not combined with; they remain computed above and are recorded
    % as the counterfactual.  beta has no part in either branch.
    if exhaustStop
        atLastLevel = hist.stage(outer) >= numel(moveLevels);
        convOuter   = mvState.ex.declared && atLastLevel;
        if mvState.ex.declared && ~atLastLevel
            % A declaration at a non-terminal level is a DESCENT, consumed by
            % olh.move.limit at the next iteration -- never a convergence.
            log{end+1} = sprintf(['iter %d: stage exhaustion declared (branch %s, ' ...
                'window %d-%d) at move %.4g; ladder descends, not converged'], ...
                outer, mvState.ex.declBranch, mvState.ex.declBegin, ...
                mvState.ex.declIter, hist.move(outer)); %#ok<AGROW>
        elseif convOuter
            log{end+1} = sprintf(['iter %d: terminal stage exhaustion declared ' ...
                '(branch %s, window %d-%d) at move %.4g, the last ladder level'], ...
                outer, mvState.ex.declBranch, mvState.ex.declBegin, ...
                mvState.ex.declIter, hist.move(outer)); %#ok<AGROW>
        end
    end

    % ---- projection continuation ----------------------------------------
    % The trigger is the EXISTING outer convergence event -- the frozen test
    % together with whatever guards are configured, i.e. the very event that
    % would otherwise have stopped the run.  No new numerical constant, no new
    % window and no new counter, and the move controller is not touched.
    projEvent = 0;
    if useProj && convOuter && projStage < numel(projLevels)
        projStage = projStage + 1;
        projEvent = 1;
        convOuter = false;
        log{end+1} = sprintf(['iter %d: outer convergence consumed by projection ' ...
            'continuation -> betaProj = %.6g at the next iteration'], ...
            outer, projLevels(projStage)); %#ok<AGROW>
    end
    hist.projEvent(outer) = projEvent;

    % ---- a p=3 problem may not be declared converged while p < 3 --------
    % Forced by the definition of the continuation experiment, not a tuning
    % choice.
    if pBlocksStop && convOuter && pNow < pSchedule(end)
        log{end+1} = sprintf(['iter %d: stop blocked, penalization p=%.4g has not ' ...
            'reached its final value %.4g'], outer, pNow, pSchedule(end)); %#ok<AGROW>
        convOuter = false;
    end

    % BENCHMARK TIMING INSTRUMENTATION ONLY.  Placed after the convergence
    % test and every guard, so the recorded outer-iteration time covers each
    % computational region of the iteration -- FE assembly, eigenproblem, modal
    % processing, sensitivities, filtering, the step controller, the nested
    % inner solve, the design update, bookkeeping and the convergence test.
    % The benchmark subtracts hist.tInner from it to obtain the outer-exclusive
    % time.  Nothing reads it back.
    hist.tOuter(outer) = toc(tOuterTic);

    if convOuter
        log{end+1} = sprintf('converged at outer iteration %d (||drho||_2 = %.3e, max|drho| = %.3e)',outer,dxNorm2,dxOuter); %#ok<AGROW>
        break
    end
end

% ---- final analysis -----------------------------------------------------
% At the terminal penalization actually in force, which equals cfg.material.
% stiffness.p whenever there is no schedule, and ALWAYS with the terminal mass
% model, so that the final formulation matches the unscheduled baseline exactly.
massFinalCfg = massCfg;  massFinalCfg.model = massTerminal;
[K,M] = assemble2D(mdl, rho, pNow, massFinalCfg);
[w, Phi, lam] = eigSolve(K, M, Jcalc, g('eigen.solver'));
T = classifyModes(mdl, M, Phi, w);

res = struct('cfg',cfg,'rho',rho,'omega',w,'lambda',lam,'hist',hist, ...
             'modeTable',T,'log',{log},'nOuter',numel(hist.N), ...
             'wallclock',toc(t0),'mdl',mdl);
res.status = local_status(log, numel(hist.N), maxOuter);
if useExhaustion
    res.exhaustion = struct( ...
        'W', mvState.ex.W, 'P', mvState.ex.P, 'Wnp', mvState.ex.Wnp, ...
        'tol', mvState.ex.tol, ...
        'stageStarts', mvState.stageStarts, ...
        'descents', mvState.descents, ...        % [iterApplied stageFrom declIter declBegin]
        'events', mvState.ex.events, ...
        'eventBranch', {mvState.ex.eventBranch}, ...
        'terminalDeclared', mvState.ex.declared, ...
        'terminalDeclIter', mvState.ex.declIter, ...
        'terminalDeclBegin', mvState.ex.declBegin, ...
        'terminalBranch', mvState.ex.declBranch, ...
        'signalDrivesMove', exhaustMove, 'ruleAdmitsStop', exhaustStop);
end
if wantDiag, res.diag = dg; end
% res.rho is ALWAYS the physical density actually used by the FE model above.
% Under projection the design variable and the filtered field are recorded
% alongside it so that no reported quantity depends on any post-hoc thresholding.
if useProj
    res.z       = z;
    res.zTilde  = (flt.H*z)./flt.Hs;
    res.rhoPhys = rho;
    res.projFinalBeta = bProj;
end
end

% =========================================================================
function [Vsum, gradV] = local_projVolume(H, Hs, z, dz, betaProj, eta, rhomin)
%LOCAL_PROJVOLUME  Exact volume and its z-gradient at the trial point z+dz.
%   Under projection V is genuinely nonlinear in z, so value and gradient are
%   formed as a consistent pair at z+dz.
[rhoT, sT] = projDensityField(H, Hs, z + dz, betaProj, eta, rhomin);
Vsum  = sum(rhoT);
gradV = projChain(H, Hs, sT, ones(numel(rhoT),1));
end

function s = local_yesno(tf)
if tf, s = 'yes'; else, s = 'NO'; end
end

function m = local_prevMove(hist, outer)
if outer >= 2, m = hist.move(outer-1); else, m = NaN; end
end

function s = local_guardName(guardLadder, guardMaxChange)
% Reported in the log for the human reader.  The historical spellings R1/R2 are
% kept in parentheses so that an old log and a new one can be compared line for
% line; the SOLVER no longer branches on them anywhere.
if guardLadder && guardMaxChange, s = 'ladderExhausted+maxDesignChange';
elseif guardLadder,               s = 'ladderExhausted (R1)';
else,                             s = 'maxDesignChange (R2)';
end
end

function s = local_status(log, nOuter, maxOuter)
%LOCAL_STATUS  Explicit status precedence.
%   SOLVER_FAILURE  >  CAP_HIT  >  CONVERGED  >  STOPPED_OTHER
%   A solver failure is reported even if the run also hit the cap, and the cap
%   outranks convergence because a capped run never satisfied the criterion.
if any(contains(log,'LP inner solve failed'))
    s = 'SOLVER_FAILURE';
elseif any(contains(log,'converged at outer iteration'))
    s = 'CONVERGED';
elseif nOuter >= maxOuter
    s = 'CAP_HIT';
else
    s = 'STOPPED_OTHER';
end
end
