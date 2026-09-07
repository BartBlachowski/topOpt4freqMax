function res = olhoffOpt(cfg)
%OLHOFFOPT  Du & Olhoff (2007) sec. 3.5 -- maximization of the n-th
%   eigenfrequency, problem (25).  Main loop of Fig. 1.
%
%   Every unstated quantity is taken from cfg and echoed into res.cfg so that
%   each figure is traceable to a config (CLAUDE.md sec.4 and sec.8).

maxNumCompThreads(cfg.threads);
t0 = tic;

if ~isfield(cfg,'mmaVariant'), cfg.mmaVariant = 'published'; end
if ~isfield(cfg,'outerNorm'),  cfg.outerNorm  = 'l2';        end
if ~isfield(cfg,'innerVar'),   cfg.innerVar   = 'drho';      end
if ~isfield(cfg,'moveFamily'), cfg.moveFamily = 'S0';        end
if ~isfield(cfg,'outerGuard'), cfg.outerGuard = 'none';      end
cfg.mmasubPath = useMMA(cfg.mmaVariant);   % recorded: which mmasub ran

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
              'innerConv',[],'cumInner',[],'dxNorm2',[],'move',[],'gap12',[], ...
              'volErr',[],'dBeta',[],'stage',[],'pPen',[],'pStage',[],'pEvent',[],'massLow',[], ...
              'projBeta',[],'projStage',[],'projEvent',[],'dxPhys2',[]);
log = {};
cumInner = 0;
% Optional per-iteration diagnostic record (WP2B of the residual-discrepancy
% audit).  PURELY ADDITIVE and default OFF: when cfg.diag is absent or false
% nothing below this flag executes and the trajectory is bit-identical.
wantDiag = isfield(cfg,'diag') && ~isempty(cfg.diag) && cfg.diag;
dg = struct('drho',{{}},'dlamPred',{{}},'lam',{{}},'N',[],'gap',[], ...
            'fdiag',[],'foff',[],'Vrot',[],'lamTild',[],'beta',[]);
mmaState = [];   % persistent MMA solver state (innerVar = 'rho' only)
multState = [];  % multiplicity-rule state (algo/multRule.m); unused by 'binary'
mvState  = [];   % step/move controller state (algo/moveControl.m)
lamPrev  = [];   % previous lambda_n, for the S3 realized-gain measure

if cfg.verbose
    fprintf('%4s %9s %9s %9s %4s %9s %6s %6s %8s %9s %8s %9s %7s\n', ...
            'it','omega1','omega2','omega3','N','sqrt(beta)', ...
            'inner','cumIn','maxdrho','|drho|2','move','vol','conv');
end

% Decoupled p-continuation state.  DEFAULT OFF: pDecoupled is false unless the
% caller sets cfg.pDecouple, so both the fixed-p realization and the previous
% ladder-coupled schedule are reproduced bitwise.
pDecoupled = isfield(cfg,'pDecouple') && ~isempty(cfg.pDecouple) && cfg.pDecouple;
if pDecoupled
    assert(isfield(cfg,'pSchedule') && ~isempty(cfg.pSchedule), ...
        'olhoffOpt:pDecouple','cfg.pDecouple requires cfg.pSchedule');
    assert(strcmpi(cfg.moveFamily,'S2'), 'olhoffOpt:pDecouple','decoupled p requires the S2 ladder');
end
pStage    = 1;    % index into cfg.pSchedule
prevStage = 1;    % S2 ladder stage seen at the previous iteration

% ---- discreteness projection (audit_m4_projection_invariance) ------------
% DEFAULT OFF.  With cfg.projection absent, empty, or .on false, every line
% guarded by useProj is skipped: the design variable IS the physical density,
% the Sigmund (1997) SENSITIVITY filter of Du & Olhoff sec.1 is in force, and
% the frozen realization is reproduced bitwise.
%
% When on, the design variable becomes z and the physical density is the
% three-field map  z -> zTilde = (H z)/Hs -> rhoPhys = rhomin+(1-rhomin)*P(.),
% with P the tanh projection.  rhoPhys is used by the stiffness assembly, the
% mass interpolation, the volume constraint and the eigenvalue evaluation, and
% ALL eigenvalue sensitivities -- diagonal AND M4 off-diagonal -- are carried
% to z by the complete chain rule (filter/projChain.m).  The filter radius,
% the move ladder, MMA, genGrad, deltaLambda and multRule are untouched.
%
% CLASS D.  Absent from Du & Olhoff (2007), the erratum, Olhoff & Du (2014)
% and the Krog & Olhoff lineage.  See WP1_RECONSTRUCTION_BOUNDARY.md; the
% chain rule is derived in WP2_PROJECTION_MATH.md.
useProj = isfield(cfg,'projection') && ~isempty(cfg.projection) ...
          && isfield(cfg.projection,'on') && cfg.projection.on;
z = [];  sChain = [];  prj = [];
if useProj
    prj = cfg.projection;
    assert(isfield(prj,'betaSchedule') && ~isempty(prj.betaSchedule), ...
        'olhoffOpt:projection','cfg.projection requires betaSchedule');
    assert(isfield(prj,'eta') && ~isempty(prj.eta), ...
        'olhoffOpt:projection','cfg.projection requires eta');
    assert(all(diff(prj.betaSchedule(:)) >= 0), 'olhoffOpt:projection', ...
        'cfg.projection.betaSchedule must be monotone non-decreasing');
    assert(strcmpi(cfg.innerVar,'drho') && ~strcmpi(cfg.innerSolver,'lp'), ...
        'olhoffOpt:projection','projection is implemented for the drho/MMA inner loop only');
    z = rho;                      % design variable, box [0,1]; rho0 uniform
end
projStage = 1;    % index into cfg.projection.betaSchedule

for outer = 1:cfg.maxOuter
    % ---- SIMP penalization continuation (audit_p_continuation) ----------
    % DEFAULT OFF.  With cfg.pSchedule absent or empty, pNow == cfg.p at every
    % iteration and this block is inert: the frozen fixed-p realization is
    % reproduced bitwise.
    %
    % When present, p is INDEXED BY THE EXISTING S2 LADDER STAGE, clamped to
    % the schedule length.  No new constant is introduced: the transition is
    % the stall event the S2 controller already computes from beta with its
    % own window and re-arm rule, and the move ladder itself is untouched.
    % mvState holds the stage established by the PREVIOUS call to moveControl,
    % which is the stage in force when this iteration assembles; before the
    % first call the stage is 1.
    pNow = cfg.p;
    if isfield(cfg,'pSchedule') && ~isempty(cfg.pSchedule)
        if pDecoupled
            % DECOUPLED (audit_m4_p_continuation_decoupled).  p advances on its
            % own counter, driven by the SAME existing beta stall event that the
            % S2 ladder consumes, but no longer sharing the ladder's index.  See
            % the interception after moveControl below.
            pNow = cfg.pSchedule(min(pStage, numel(cfg.pSchedule)));
        else
            % COUPLED (audit_p_continuation): p indexed by the S2 ladder stage.
            if isempty(mvState) || ~isfield(mvState,'stage'); stageInForce = 1;
            else;                                            stageInForce = mvState.stage; end
            pNow = cfg.pSchedule(min(stageInForce, numel(cfg.pSchedule)));
        end
    end
    hist.pPen(outer)   = pNow;
    hist.pStage(outer) = pStage;

    % ---- printed-mass-model continuation (audit_pm1) --------------------
    % DEFAULT OFF.  With cfg.massLowP absent the terminal cfg.massInterp is
    % used at every iteration and this block is inert, so both the frozen
    % realization and the P1 / PD1 schedules reproduce bitwise.
    %
    % When present, the PRINTED alternative model cfg.massLowP is in force
    % while p is below its final value, and the terminal model cfg.massInterp
    % resumes exactly when p first reaches that value.  The switch introduces
    % no numerical threshold of its own: it is tied to the existing p schedule.
    massNow = cfg.massInterp;
    if isfield(cfg,'massLowP') && ~isempty(cfg.massLowP) ...
            && isfield(cfg,'pSchedule') && ~isempty(cfg.pSchedule) ...
            && pNow < cfg.pSchedule(end)
        massNow = cfg.massLowP;
    end
    hist.massLow(outer) = ~strcmpi(massNow, cfg.massInterp);

    % ---- projection sharpness in force this iteration -------------------
    % Inert when useProj is false: bProj stays 0 and rho is untouched.
    bProj = 0;
    if useProj
        bProj = prj.betaSchedule(min(projStage, numel(prj.betaSchedule)));
        [rho, sChain] = projDensityField(flt.H, flt.Hs, z, bProj, prj.eta, cfg.rhomin);
    end
    hist.projBeta(outer)  = bProj;
    hist.projStage(outer) = projStage;

    % ---- step 1: FE analysis + multiplicity detection -------------------
    te = tic;
    [K,M] = assemble2D(mdl, rho, pNow, massNow);
    [w, Phi, lam] = eigSolve(K, M, Jcalc, cfg.solver);
    tEig = toc(te);

    % Multiplicity treatment.  algo/multRule.m holds the rule and its A/B/C
    % evidence classification; cfg.multRule = 'binary' (default) is the frozen
    % memoryless test and reproduces the previous trajectory bitwise.
    [N, multState] = multRule(cfg, w, n, Jcalc, multState);
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
    % Sec. 3.5.1: "In the second step of the main loop, we set lambda~ = omega_n^2"
    % -- the FIRST eigenvalue of the cluster, NOT the cluster mean.  (Identical
    % at exact degeneracy; differs by up to tolMult at the detection threshold.)
    lamTild = lam(n);
    F       = genGrad(mdl, rho, pNow, massNow, Phi, lamTild, idx);
    % Diagonal-offset ('subspace') form: the subeigenvalue problem keeps the
    % actual separation lam(j)-lam(n), so the diagonal blocks must use each
    % mode's OWN eigenvalue -- which is eq. (24) verbatim, f_jj with lambda_j.
    % The off-diagonals keep (19)'s lambda-tilde.  At exact degeneracy the two
    % coincide and the whole thing collapses onto (25d) as printed.
    useOff = isfield(cfg,'multRule') && strcmpi(cfg.multRule,'subspace');
    if useOff
        for j = 1:N
            Gj = genGrad(mdl, rho, pNow, massNow, Phi, lam(idx(j)), idx(j));
            F(:,j,j) = Gj(:,1,1);
        end
        dOff = lam(idx) - lam(idx(1));
    else
        dOff = [];
    end
    FJ      = genGrad(mdl, rho, pNow, massNow, Phi, lam(J), J);
    fJJ     = FJ(:,1,1);

    % ---- sensitivity treatment ------------------------------------------
    if useProj
        % Complete chain rule  d/dz = W' S d/drho  (WP2 sec.3.2), applied to
        % EVERY generalized gradient -- diagonal AND off-diagonal -- and to
        % f_JJ.  a_sk is a scalar function of rho, so the chain rule does not
        % distinguish s==k from s~=k.  The frozen realization uses
        % filterMode='all', i.e. it already filters every f_sk and f_JJ, so the
        % two paths touch the SAME set of gradients and differ only in the
        % operator applied; cfg.filterMode is therefore not consulted here.
        % The transform precedes deltaLambda so that the
        % sub-eigenvalue matrix A, its eigenvectors and its gradients are all
        % expressed in z (WP2 sec.3.3, requirement 7).
        %
        % This REPLACES the Sigmund (1997) sensitivity filter.  Class-D item
        % D2/D4: Du & Olhoff sec.1 states the filter was applied to the
        % sensitivities, so this is a departure from a printed choice.
        for s = 1:N
            for k = s:N
                v = projChain(flt.H, flt.Hs, sChain, F(:,s,k));
                F(:,s,k) = v;  F(:,k,s) = v;
            end
        end
        fJJ  = projChain(flt.H, flt.Hs, sChain, fJJ);
    else
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
    end
    tGrad = toc(tg);

    % ---- step/move control (algo/moveControl.m; see its header for the
    % A/B/C evidence classification -- NONE of this is specified by the paper)
    if ~isempty(lamPrev)
        mvState.lastRealized = lam(n) - lamPrev;
    elseif ~isempty(mvState)
        mvState.lastRealized = NaN;
    end
    [mvNow, mvState] = moveControl(cfg, outer, hist, mvState);
    % ---- decoupled p continuation: intercept the stall event ------------
    % moveControl is left byte-frozen.  A stall is detected here as an S2 stage
    % increase.  While p has not reached its final value that event is consumed
    % by the p controller INSTEAD of the ladder: p advances one step, the move
    % is restored to the ladder's own first level, and the ladder index and its
    % re-arm clock are reset so the existing >W rule supplies the dwell.  Once
    % p is final the event is left to the ladder and refinement proceeds
    % normally.  No new numerical constant is introduced.
    pEvent = 0;
    if pDecoupled
        stageNow = 1; if isfield(mvState,'stage'), stageNow = mvState.stage; end
        if stageNow > prevStage && pStage < numel(cfg.pSchedule)
            pStage        = pStage + 1;
            mvState.stage = 1;
            mvState.lastStage = outer;
            mvNow         = cfg.s2Levels(1);
            pEvent        = 1;
            log{end+1} = sprintf(['iter %d: stall consumed by p continuation -> p=%.4g ' ...
                'at next iteration; move restored to %.4g, S2 stage reset to 1'], ...
                outer, cfg.pSchedule(pStage), mvNow); %#ok<AGROW>
        end
        stageNow = 1; if isfield(mvState,'stage'), stageNow = mvState.stage; end
        prevStage = stageNow;
    end
    hist.pEvent(outer) = pEvent;
    lamPrev = lam(n);

    % ---- step 3: inner loop ---------------------------------------------
    ti = tic;
    ctx = struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J), ...
                 'rho',rho,'rhomin',cfg.rhomin,'volfrac',cfg.volfrac, ...
                 'move',mvNow,'maxInner',cfg.maxInner, ...
                 'tolInner',cfg.tolInner,'minInner',cfg.minInner, ...
                 'offDiag',cfg.offDiag,'dOff',dOff);
    if useProj
        % The optimization variable is z, so the MMA box and the move ladder
        % bound dz:  max(0-z,-move) <= dz <= min(1-z,+move)  (WP2 sec.4.1).
        % Passing z through ctx.rho and 0 through ctx.rhomin reuses the frozen
        % box code verbatim.  The volume constraint (25e) is evaluated exactly
        % at z+dz through volFun (WP2 sec.4.3).
        ctx.rho    = z;
        ctx.rhomin = 0;
        ctx.volFun = @(dz) projVolume(flt.H, flt.Hs, z, dz, bProj, prj.eta, cfg.rhomin);
    end
    if strcmpi(cfg.innerSolver,'lp')
        [drho, st] = innerLoopLP(ctx);
        if ~st.conv
            log{end+1} = sprintf('iter %d: LP inner solve failed (flag=%d)',outer,st.lpFlag); %#ok<AGROW>
        end
    elseif strcmpi(cfg.innerVar,'rho')
        [drho, st, mmaState] = innerLoopRho(ctx, mmaState);
    else
        [drho, st] = innerLoop(ctx);
    end
    tInner = toc(ti);

    % ---- step 4: update --------------------------------------------------
    % Under projection the update is applied to the OPTIMIZATION variable z
    % (drho holds dz), and the physical density is recomputed from the map so
    % that hist.vol below records the post-update PHYSICAL volume, exactly as
    % in the frozen semantics.
    if useProj
        z   = min(1, max(0, z + drho));
        rhoPrev = rho;
        [rho, sChain] = projDensityField(flt.H, flt.Hs, z, bProj, prj.eta, cfg.rhomin);
        hist.dxPhys2(outer) = norm(rho - rhoPrev);   % cross-reference only
    else
        rho = min(1, max(cfg.rhomin, rho + drho));
        hist.dxPhys2(outer) = NaN;
    end
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
    hist.volErr(outer)   = mean(rho) - cfg.volfrac;
    if outer > 1, hist.dBeta(outer) = st.beta - hist.beta(outer-1);
    else,         hist.dBeta(outer) = NaN; end
    if isfield(mvState,'stage'), hist.stage(outer) = mvState.stage;
    else,                        hist.stage(outer) = 1; end

    if cfg.verbose
        fprintf('%4d %9.2f %9.2f %9.2f %4d %9.2f %6d %6d %8.4f %9.4f %8.4f %9.3f %7s\n', ...
                outer, w(1), w(2), w(min(3,end)), N, sqrt(max(st.beta,0)), ...
                st.nInner, cumInner, dxOuter, dxNorm2, mvNow, mean(rho), ...
                stringYesNo(st.conv));
    end

    % Fig. 1 tests "||drho|| < eps" -- a vector NORM, read here as the
    % Euclidean norm (the natural reading of the unqualified symbol).  The
    % max-norm is kept as a labelled alternative.  epsilon itself is unstated
    % (CLAUDE.md sec.4) and is a per-run recorded parameter.
    switch lower(cfg.outerNorm)
        case 'l2',  convOuter = dxNorm2 < cfg.tolOuter;
        case 'max', convOuter = dxOuter < cfg.tolOuter;
        otherwise,  error('olhoffOpt:outerNorm','unknown outerNorm %s',cfg.outerNorm);
    end
    % Guard on WHEN that test may be believed.  The paper places no bound on
    % drho other than the box (25f), so ||drho|| -> 0 genuinely means the design
    % has stopped moving.  This reconstruction adds a move ladder, and then
    % ||drho||_inf <= mv_k: on an iteration where mv_k differs from mv_{k-1} a
    % scheduled reduction of the move limit mechanically reduces the measured
    % step, with no change in the design's behaviour.  The criterion is
    % uninterpretable there.  ('none' = legacy, bitwise-identical trajectories.)
    switch lower(cfg.outerGuard)
        case 'none'
            % legacy: evaluate every iteration
        case 'settledmove'
            moveSettled = outer >= 2 && hist.move(outer) == hist.move(outer-1);
            if convOuter && ~moveSettled
                log{end+1} = sprintf(['iter %d: ||drho|| below eps but the move limit ' ...
                    'just changed (%.6g -> %.6g); convergence NOT asserted'], ...
                    outer, mvPrevForLog(hist,outer), hist.move(outer)); %#ok<AGROW>
            end
            convOuter = convOuter && moveSettled;
        otherwise
            error('olhoffOpt:outerGuard','unknown outerGuard %s',cfg.outerGuard);
    end
    % Topology-restoration audit: optional, preregistered stopping safeguards.
    % Neither branch changes the update or S2 controller. Absent field keeps
    % the frozen M4 rule; see audit_m4_topology_restoration/PREREGISTRATION.md.
    if isfield(cfg,'restorationGuard') && ~isempty(cfg.restorationGuard)
        epsRMS = cfg.tolOuter/sqrt(NE);
        switch upper(cfg.restorationGuard)
            case 'R1'
                assert(strcmpi(cfg.moveFamily,'S2'), 'R1 requires the S2 ladder');
                restorationReady = ~any(cfg.s2Levels(hist.stage(outer)+1:end) > epsRMS);
            case 'R2'
                restorationReady = dxOuter < epsRMS;
            otherwise
                error('olhoffOpt:restorationGuard', 'Unknown restoration guard %s',cfg.restorationGuard);
        end
        if convOuter && ~restorationReady
            log{end+1} = sprintf('iter %d: baseline stop blocked by %s (stage=%d, max|drho|=%.6g, epsRMS=%.6g)', ...
                outer,cfg.restorationGuard,hist.stage(outer),dxOuter,epsRMS); %#ok<AGROW>
        end
        convOuter = convOuter && restorationReady;
    end
    % ---- projection continuation ----------------------------------------
    % The trigger is the EXISTING outer convergence event -- the frozen test
    % together with whatever guards are configured, i.e. the very event that
    % would otherwise have stopped the run.  No new numerical constant, no new
    % window and no new counter are introduced, and the S2 move ladder is not
    % touched.  This mirrors the pSchedule stop-block immediately below and is
    % the standard "converge, sharpen, re-converge" projection continuation.
    % Inert when useProj is false.
    projEvent = 0;
    if useProj && convOuter && projStage < numel(prj.betaSchedule)
        projStage = projStage + 1;
        projEvent = 1;
        convOuter = false;
        log{end+1} = sprintf(['iter %d: outer convergence consumed by projection ' ...
            'continuation -> betaProj = %.6g at the next iteration'], ...
            outer, prj.betaSchedule(projStage)); %#ok<AGROW>
    end
    hist.projEvent(outer) = projEvent;
    % A p=3 problem cannot be declared converged while p is still below 3.
    % Forced by the definition of the continuation experiment, not a tuning
    % choice; inert when cfg.pSchedule is absent.
    if isfield(cfg,'pSchedule') && ~isempty(cfg.pSchedule) && convOuter ...
            && pNow < cfg.pSchedule(end)
        log{end+1} = sprintf(['iter %d: stop blocked, penalization p=%.4g has not ' ...
            'reached its final value %.4g'], outer, pNow, cfg.pSchedule(end)); %#ok<AGROW>
        convOuter = false;
    end
    if convOuter
        log{end+1} = sprintf('converged at outer iteration %d (||drho||_2 = %.3e, max|drho| = %.3e)',outer,dxNorm2,dxOuter); %#ok<AGROW>
        break
    end
end

% ---- final analysis -----------------------------------------------------
% At the terminal penalization actually in force, which equals cfg.p whenever
% the schedule is absent, and ALWAYS with the terminal mass model cfg.massInterp
% (PM1 requires the final formulation to match the frozen baseline exactly).
[K,M] = assemble2D(mdl, rho, pNow, cfg.massInterp);
[w, Phi, lam] = eigSolve(K, M, Jcalc, cfg.solver);
T = classifyModes(mdl, M, Phi, w);

res = struct('cfg',cfg,'rho',rho,'omega',w,'lambda',lam,'hist',hist, ...
             'modeTable',T,'log',{log},'nOuter',numel(hist.N), ...
             'wallclock',toc(t0),'mdl',mdl);
if wantDiag, res.diag = dg; end
% res.rho is ALWAYS the physical density actually used by the FE model above.
% Under projection the unprojected and filtered fields are recorded alongside
% it so that no reported quantity depends on any post-hoc thresholding.
if useProj
    res.z       = z;
    res.zTilde  = (flt.H*z)./flt.Hs;
    res.rhoPhys = rho;
    res.projFinalBeta = bProj;
end
end

function [Vsum, gradV] = projVolume(H, Hs, z, dz, betaProj, eta, rhomin)
%PROJVOLUME  Exact volume and its z-gradient at the trial point z+dz.
%   Mirrors the frozen code's EXACT evaluation of (25e) at each MMA
%   sub-iterate; under projection V is genuinely nonlinear in z, so value and
%   gradient are formed as a consistent pair at z+dz (WP2 sec.4.3).
[rhoT, sT] = projDensityField(H, Hs, z + dz, betaProj, eta, rhomin);
Vsum  = sum(rhoT);
gradV = projChain(H, Hs, sT, ones(numel(rhoT),1));
end

function s = stringYesNo(tf)
if tf, s = 'yes'; else, s = 'NO'; end
end

function m = mvPrevForLog(hist, outer)
if outer >= 2, m = hist.move(outer-1); else, m = NaN; end
end
