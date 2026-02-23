% A TOPOLOGY OPTIMIZATION CODE FOR FREQUENCY MAXIMIZATION
% Rewritten from the Python version (Aage & Johansen, 2013, modified)
%
% Supports aggregated compliance over multiple load cases in runCfg.load_cases.
% Includes semi_harmonic loads with cached baseline (M0, Phi0, omega0).
% Legacy fallback remains the original fixed harmonic load behavior when
% runCfg.load_cases is not provided.

function [xOut, fHz, tIter, nIter, info] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, varargin)
    xOut = [];
    fHz = NaN(3, 1);
    tIter = NaN;
    nIter = NaN;
    info = struct();

    if nargin < 8
        nelx = 240; nely = 30; volfrac = 0.4; penal = 3.0;
        rmin = 0.05; ft = 0; L = 8.0; H = 1.0;
    end
    if nargin >= 9 && ~isempty(varargin{1})
        runCfg = varargin{1};
        if ~isstruct(runCfg)
            error('Optional 9th input must be a struct when provided.');
        end
    else
        runCfg = struct();
    end
    localEnsurePlotHelpersOnPath();

    fprintf('Compliance objective with load-case aggregation\n');
    fprintf('mesh: %d x %d\n', nelx, nely);
    fprintf('domain: L x H = %g x %g\n', L, H);
    fprintf('volfrac: %g, rmin(phys): %g, penal: %g\n', volfrac, rmin, penal);
    ftnames = {'Sensitivity based', 'Density based'};
    fprintf('Filter method: %s\n', ftnames{ft+1});

    if L <= 0 || H <= 0
        error('L and H must be positive.');
    end
    hx = L / nelx;
    hy = H / nely;
    fprintf('element size: hx=%g, hy=%g\n', hx, hy);

    % --- Material / SIMP ---
    Emax = localOpt(runCfg, 'E0', 1e7);
    Emin = localOpt(runCfg, 'Emin', 1e-2);
    rho_min = localOpt(runCfg, 'rho_min', 1e-6);
    rho0 = localOpt(runCfg, 'rho0', 1.0);
    pmass = localOpt(runCfg, 'pmass', 1.0);
    nu = localOpt(runCfg, 'nu', 0.3);
    move = localOpt(runCfg, 'move', 0.2);
    convTol = localOpt(runCfg, 'conv_tol', 0.01);
    maxIters = localOpt(runCfg, 'max_iters', 2000);
    supportType = upper(string(localOpt(runCfg, 'supportType', "SS")));
    approachName = localApproachName(runCfg, 'ourApproach');
    optimizerType = upper(strtrim(string(localOpt(runCfg, 'optimizer', 'OC'))));
    if ~any(strcmp(optimizerType, {'OC','MMA'}))
        error('topopt_freq:InvalidOptimizer', ...
            'runCfg.optimizer must be "OC" or "MMA" (got "%s").', optimizerType);
    end
    fprintf('Optimizer: %s\n', optimizerType);
    if isfield(runCfg, 'visualise_live') && ~isempty(runCfg.visualise_live)
        visualiseLive = localParseVisualiseLive(runCfg.visualise_live, true);
    else
        visualiseLive = true;
    end
    visualizationQuality = localParseVisualizationQuality( ...
        localOpt(runCfg, 'visualization_quality', 'regular'));
    saveFrqIterations = localParseVisualiseLive(localOpt(runCfg, 'save_frq_iterations', false), false);
    harmonicNormalize = localParseVisualiseLive(localOpt(runCfg, 'harmonic_normalize', true), true);
    debugReturnDc = localParseVisualiseLive(localOpt(runCfg, 'debug_return_dc', false), false);
    debugSemiHarmonic = localParseVisualiseLive(localOpt(runCfg, 'debug_semi_harmonic', false), false);
    semiHarmonicBaseline = localParseSemiHarmonicBaseline( ...
        localOpt(runCfg, 'semi_harmonic_baseline', 'solid'));
    semiHarmonicRhoSource = localParseSemiHarmonicRhoSource( ...
        localOpt(runCfg, 'semi_harmonic_rho_source', 'x'));

    % Passive element sets (forced density = 1 or 0, excluded from OC update).
    nEl = nelx * nely;
    pasS = []; pasV = [];
    if isfield(runCfg, 'pasS') && ~isempty(runCfg.pasS), pasS = runCfg.pasS(:); end
    if isfield(runCfg, 'pasV') && ~isempty(runCfg.pasV), pasV = runCfg.pasV(:); end
    act = setdiff((1:nEl)', union(pasS, pasV));
    if saveFrqIterations
        fprintf(['Warning: save_frq_iterations=yes forces per-iteration eigenvalue solves for plotting; ', ...
            'runtime will increase and comparisons are not fair.\n']);
        info.freq_iter_omega = NaN(maxIters, 3);
    end

    ndof = 2*(nelx+1)*(nely+1);
    nNodes = (nelx+1) * (nely+1);
    nodeIds = (1:nNodes)';
    nodeIdx0 = nodeIds - 1;
    nodeX = floor(nodeIdx0 / (nely + 1)) * hx;
    nodeY = mod(nodeIdx0, (nely + 1)) * hy;
    yDown = zeros(ndof, 1);
    yDown(2:2:end) = -1;

    [loadCases, usingConfiguredLoadCases, maxHarmonicMode, modeUpdateAfter, maxSemiHarmonicMode] = ...
        localResolveLoadCases(runCfg, nodeX, nodeY, nodeIds);
    nCases = numel(loadCases);
    fprintf('Load cases: %d\n', nCases);
    for ci = 1:nCases
        fprintf('  case[%d] \"%s\": factor=%g, nLoads=%d\n', ...
            ci, loadCases(ci).name, loadCases(ci).factor, numel(loadCases(ci).loads));
    end
    hasSemiHarmonicLoads = usingConfiguredLoadCases && maxSemiHarmonicMode > 0;
    semiDebugIters = [1, 10];
    if hasSemiHarmonicLoads
        fprintf('[Load cases] semi_harmonic baseline=%s, rho_source=%s\n', ...
            semiHarmonicBaseline, semiHarmonicRhoSource);
    end

    % Allocate design variables (passive elements pinned; active adjusted for volfrac).
    x = volfrac * ones(nEl, 1);
    x(pasS) = 1;
    x(pasV) = 0;
    if ~isempty(pasS) || ~isempty(pasV)
        nact = numel(act);
        if nact > 0
            act_target = (volfrac * nEl - numel(pasS)) / nact;
            x(act) = min(1, max(0, act_target));
        end
    end
    xold  = x;
    xPhys = x;
    g     = 0;

    % MMA persistent state (only used when optimizerType='MMA').
    % Sized to active elements only: passive elements (pasS/pasV) are excluded
    % because they have xmin==xmax, which makes mmasub's xl1/ux1==0 → division
    % by zero → RCOND=NaN.  Active set 'act' is fixed for the whole run.
    if strcmp(optimizerType, 'MMA')
        n_act     = numel(act);
        mma_low   = zeros(n_act, 1);
        mma_upp   = ones(n_act, 1);
        mma_xold1 = x(act);
        mma_xold2 = x(act);
    end

    % FE: element stiffness & mass matrices
    KE = lk(hx, hy, nu);
    ME = lm(hx, hy);

    % Build edofMat using standard Q4 connectivity (LL, LR, UR, UL).
    % Previous mapping mixed ordering/offsets and caused inconsistent assembly.
    edofMat = zeros(nelx*nely, 8);
    for elx = 0:nelx-1
        for ely = 0:nely-1
            el  = ely + elx*nely + 1;  % 1-based element index
            n1  = (nely+1)*elx + ely;      % LL (0-based)
            n2  = (nely+1)*(elx+1) + ely;  % LR
            n3  = n2 + 1;                  % UR
            n4  = n1 + 1;                  % UL
            edofMat(el,:) = [2*n1+1, 2*n1+2, ...
                             2*n2+1, 2*n2+2, ...
                             2*n3+1, 2*n3+2, ...
                             2*n4+1, 2*n4+2];
        end
    end

    % Index vectors for sparse assembly
    iK = reshape(kron(edofMat, ones(1,8))', [], 1);
    jK = reshape(kron(edofMat, ones(8,1))', [], 1);

    % Filter: build sparse filter matrix
    rminx = max(1, ceil(rmin/hx));
    rminy = max(1, ceil(rmin/hy));
    nfilter = nelx*nely*(2*(rminx-1)+1)*(2*(rminy-1)+1);
    iH = zeros(nfilter, 1);
    jH = zeros(nfilter, 1);
    sH = zeros(nfilter, 1);
    cc = 0;
    for i = 0:nelx-1
        for j = 0:nely-1
            row = i*nely + j + 1;  % 1-based
            kk1 = max(i-(rminx-1), 0);
            kk2 = min(i+rminx, nelx) - 1;
            ll1 = max(j-(rminy-1), 0);
            ll2 = min(j+rminy, nely) - 1;
            for k = kk1:kk2
                for l = ll1:ll2
                    cc = cc + 1;
                    col = k*nely + l + 1;  % 1-based
                    dx = (i-k)*hx;
                    dy = (j-l)*hy;
                    fac = rmin - sqrt(dx*dx + dy*dy);
                    iH(cc) = row;
                    jH(cc) = col;
                    sH(cc) = max(0.0, fac);
                end
            end
        end
    end
    Hf = sparse(iH(1:cc), jH(1:cc), sH(1:cc), nelx*nely, nelx*nely);
    Hs = sum(Hf, 2);
    clear iH jH sH;

    % Convert support type to constrained DOFs (1-based indexing).
    fixed = localBuildFixedDofs(supportType, nelx, nely);
    if isfield(runCfg, 'extraFixedDofs') && ~isempty(runCfg.extraFixedDofs)
        fixed = unique([fixed(:); runCfg.extraFixedDofs(:)]);
    end
    alldofs = 1:ndof;
    free = setdiff(alldofs, fixed);

    F = zeros(ndof, nCases);
    U = zeros(ndof, nCases);
    obj = NaN;
    objCases = NaN(nCases, 1);

    omegaLegacy = NaN;
    PhiLegacy = zeros(ndof, 1);
    nodalProjectionCache = [];
    semiHarmonicM0 = sparse(ndof, ndof);
    semiHarmonicOmega0 = NaN(max(maxSemiHarmonicMode, 0), 1);
    semiHarmonicPhi0 = zeros(ndof, max(maxSemiHarmonicMode, 0));
    semiHarmonicBaseVec = zeros(ndof, max(maxSemiHarmonicMode, 0));
    semiHarmonicBaselineInfo = struct( ...
        'kind', '', ...
        'rhoSource', '', ...
        'nEl', nEl, ...
        'xBaseMean', NaN, ...
        'xBaseMin', NaN, ...
        'xBaseMax', NaN, ...
        'passiveSolidCount', numel(pasS), ...
        'passiveVoidCount', numel(pasV));
    if hasSemiHarmonicLoads
        rhoSourceInit = localSemiHarmonicRhoSourceVector(semiHarmonicRhoSource, x, xPhys);
        [~, nodalProjectionCache] = projectQ4ElementDensityToNodes(rhoSourceInit, nelx, nely);
    end

    if ~usingConfiguredLoadCases
        % Legacy fallback (no runCfg.load_cases): preserve previous behavior.
        sK0 = reshape(KE(:) * (Emin + xPhys'.^penal * (Emax - Emin)), [], 1);
        K0  = sparse(iK, jK, sK0, ndof, ndof);
        K0  = (K0 + K0') / 2;

        rhoPhys0 = rho_min + xPhys.^pmass * (rho0 - rho_min);
        sM0 = reshape(ME(:) * rhoPhys0', [], 1);
        M0  = sparse(iK, jK, sM0, ndof, ndof);
        M0  = (M0 + M0') / 2;

        K0f = K0(free, free);
        M0f = M0(free, free);
        clear K0 sK0 M0 sM0 rhoPhys0;

        [Phi_free, Lam] = eigs(K0f, M0f, 2, 'smallestabs');
        lam_vals = diag(Lam);
        [lam1, idx] = min(lam_vals);
        omegaLegacy = sqrt(max(lam1, 0));

        phi1_free = Phi_free(:, idx);
        mn = phi1_free' * (M0f * phi1_free);
        if mn > 0
            phi1_free = phi1_free / sqrt(mn);
        end

        PhiLegacy(free) = phi1_free;
        clear Phi_free Lam phi1_free K0f M0f;
        fprintf('[Eigen] lambda1=%.6e, omega1=%.6e rad/s (computed once, fixed)\n', lam1, omegaLegacy);
    else
        if maxHarmonicMode > 0
            schedParts = cell(maxHarmonicMode, 1);
            for k = 1:maxHarmonicMode
                ua = modeUpdateAfter(k);
                if ua == 0
                    schedParts{k} = sprintf('mode%d(frozen,ua=0)', k);
                else
                    schedParts{k} = sprintf('mode%d(every %d it,ua=%d)', k, ua, ua);
                end
            end
            fprintf('[Load cases] Harmonic update schedule: %s\n', strjoin(schedParts, ', '));
            fprintf('[Load cases] Frozen-eigenpair sensitivity: d(omega)/drho and d(Phi)/drho ignored.\n');
        end

        if hasSemiHarmonicLoads
            % semi_harmonic uses one fixed baseline model:
            %   F_semi(x) = rho_nodal(x) .* (omega0_k * (M0 * Phi0_k))
            % Fix: baseline is explicit/configurable (solid by default), not
            % implicitly the initial xPhys (volfrac) field.
            xBase = localBuildSemiHarmonicBaseline( ...
                semiHarmonicBaseline, xPhys, pasS, pasV, nEl);
            semiHarmonicBaselineInfo.kind = semiHarmonicBaseline;
            semiHarmonicBaselineInfo.rhoSource = semiHarmonicRhoSource;
            semiHarmonicBaselineInfo.xBaseMean = mean(xBase);
            semiHarmonicBaselineInfo.xBaseMin = min(xBase);
            semiHarmonicBaselineInfo.xBaseMax = max(xBase);

            % Baseline matrices are intentionally built from xBase, not current xPhys.
            sK0 = reshape(KE(:) * (Emin + xBase'.^penal * (Emax - Emin)), [], 1);
            K0  = sparse(iK, jK, sK0, ndof, ndof);
            K0  = (K0 + K0') / 2;

            rhoPhys0 = rho_min + xBase.^pmass * (rho0 - rho_min);
            sM0 = reshape(ME(:) * rhoPhys0', [], 1);
            semiHarmonicM0 = sparse(iK, jK, sM0, ndof, ndof);
            semiHarmonicM0 = (semiHarmonicM0 + semiHarmonicM0') / 2;

            [semiHarmonicOmega0, semiHarmonicPhi0] = localCurrentModesFromSubmatrices( ...
                K0(free, free), semiHarmonicM0(free, free), free, ndof, maxSemiHarmonicMode);
            for k = 1:maxSemiHarmonicMode
                if ~isfinite(semiHarmonicOmega0(k))
                    error('topopt_freq:SemiHarmonicModeUnavailable', ...
                        'Unable to evaluate semi_harmonic mode %d from baseline model.', k);
                end
                semiHarmonicBaseVec(:, k) = semiHarmonicOmega0(k) * (semiHarmonicM0 * semiHarmonicPhi0(:, k));
            end
            clear K0 sK0 sM0 rhoPhys0;
            fprintf('[Load cases] semi_harmonic baseline cached up to mode %d.\n', maxSemiHarmonicMode);
        end
    end

    % ------------------------------------------------------------------
    % Eigenpair cache for harmonic loads (persists across iterations).
    % Updated only when modes are due per modeUpdateAfter schedule.
    % ------------------------------------------------------------------
    harmonicOmegasCache = NaN(max(maxHarmonicMode, 0), 1);
    harmonicPhiCache    = zeros(ndof, max(maxHarmonicMode, 0));
    harmonicNormRefCache = NaN(max(maxHarmonicMode, 0), 1);
    if usingConfiguredLoadCases && maxHarmonicMode > 0 && harmonicNormalize
        fprintf('[Load cases] Harmonic RHS norm normalization: ON (per mode, anchored at first evaluation)\n');
    end

    % ------------------------------------------------------------------
    % Optimization loop
    % ------------------------------------------------------------------
    loop   = 0;
    change = 1;
    dv = ones(nelx*nely, 1);
    dc = ones(nelx*nely, 1);
    loop_tic = tic;

    while change > convTol && loop < maxIters
        loop = loop + 1;
        mmaConstraintPre = NaN;
        mmaConstraintPost = NaN;
        mmaProjected = false;

        % Assemble M(x); needed for rho-dependent loads and (optional) mode solves.
        rhoPhys = rho_min + xPhys.^pmass * (rho0 - rho_min);
        sM = reshape(ME(:) * rhoPhys', [], 1);
        M  = sparse(iK, jK, sM, ndof, ndof);
        M  = (M + M') / 2;
        clear sM;

        % Assemble K(x) and solve FE for all load cases at once.
        sK = reshape(KE(:) * (Emin + xPhys'.^penal * (Emax - Emin)), [], 1);
        K  = sparse(iK, jK, sK, ndof, ndof);
        K  = (K + K') / 2;
        clear sK;

        Kf = K(free, free);

        % Determine which harmonic modes are due for eigenpair update this iteration.
        if usingConfiguredLoadCases && maxHarmonicMode > 0
            dueFlags = false(maxHarmonicMode, 1);
            for k = 1:maxHarmonicMode
                ua = modeUpdateAfter(k);
                if ua == 0
                    dueFlags(k) = (loop == 1);   % frozen: compute once at it=1
                else
                    dueFlags(k) = (mod(loop - 1, ua) == 0);  % periodic
                end
            end
            dueModes = find(dueFlags);
        else
            dueModes = [];
        end

        needMf = saveFrqIterations || ~isempty(dueModes);
        if needMf
            Mf = M(free, free);
        else
            Mf = [];
        end

        % Update eigenpair cache for all due modes (single eigs call up to max needed).
        if ~isempty(dueModes)
            maxModeNeeded = max(dueModes);
            [newOmegas, newPhi] = localCurrentModesFromSubmatrices( ...
                Kf, Mf, free, ndof, maxModeNeeded);
            for k = 1:maxModeNeeded
                if isfinite(newOmegas(k))
                    harmonicOmegasCache(k) = newOmegas(k);
                    harmonicPhiCache(:, k)  = newPhi(:, k);
                end
            end
            % Diagnostic: separate frozen (ua=0) from periodic (ua>0) modes.
            frozenDue   = dueModes(modeUpdateAfter(dueModes) == 0);
            periodicDue = dueModes(modeUpdateAfter(dueModes) >  0);
            if ~isempty(frozenDue)
                fprintf('[Harmonic update] it=%d mode<=%d computed once (update_after=0)\n', ...
                    loop, max(frozenDue));
            end
            if ~isempty(periodicDue)
                uaVals = unique(modeUpdateAfter(periodicDue));
                for uaVal = uaVals(:)'
                    modesThisUa = periodicDue(modeUpdateAfter(periodicDue) == uaVal);
                    fprintf('[Harmonic update] it=%d mode<=%d recomputed (update_after=%d)\n', ...
                        loop, max(modesThisUa), uaVal);
                end
            end
        end

        % Expose cache as local variables for load assembly and sensitivity.
        harmonicOmegas = harmonicOmegasCache;
        harmonicPhi    = harmonicPhiCache;

        if hasSemiHarmonicLoads
            % rho_nodal source is explicit and shared with the same Pavg map
            % used by semi_harmonic sensitivity.
            rhoSourceVec = localSemiHarmonicRhoSourceVector(semiHarmonicRhoSource, x, xPhys);
            [rhoNodal, nodalProjectionCache] = projectQ4ElementDensityToNodes( ...
                rhoSourceVec, nelx, nely, nodalProjectionCache);
        else
            rhoSourceVec = [];
            rhoNodal = [];
        end

        % Debug verification (opt-in): run only at iterations 1 and 10.
        debugSemiThisIter = debugSemiHarmonic && hasSemiHarmonicLoads && any(loop == semiDebugIters);
        [F, caseDiag, harmonicNormRefCache, semiDebugAssembly] = localBuildLoadMatrix( ...
            loadCases, ndof, M, yDown, harmonicPhi, harmonicOmegas, PhiLegacy, omegaLegacy, ...
            harmonicNormRefCache, harmonicNormalize, rhoNodal, semiHarmonicBaseVec, semiHarmonicOmega0, ...
            debugSemiThisIter);

        U(:) = 0;
        U(free,:) = Kf \ F(free,:);
        if saveFrqIterations
            if isempty(Mf)
                Mf = M(free, free);
            end
            info.freq_iter_omega(loop,:) = localFirstNOmegasFromSubmatrices(Kf, Mf, 3);
        end

        if debugSemiThisIter
            localDebugSemiHarmonicIteration( ...
                loop, loadCases, semiDebugAssembly, ndof, free, Kf, ...
                rhoSourceVec, rhoNodal, nelx, nely, nodalProjectionCache, ...
                semiHarmonicBaselineInfo, semiHarmonicM0, semiHarmonicPhi0, semiHarmonicOmega0, ...
                semiHarmonicBaseVec, act);
        end
        clear Kf Mf;

        % Objective and sensitivities: sum contributions over load cases.
        obj = 0.0;
        objCases(:) = 0.0;
        dc(:) = 0.0;
        stiffScale = -penal * xPhys.^(penal-1) * (Emax - Emin);
        dMdxScale = pmass * (xPhys.^(pmass-1)) * (rho0 - rho_min);

        for icase = 1:nCases
            Ui = U(:, icase);
            objCase = Ui' * (K * Ui);
            objCases(icase) = objCase;
            obj = obj + objCase;

            ue = Ui(edofMat);  % nelx*nely x 8
            ce = sum((ue * KE) .* ue, 2);
            dc = dc + stiffScale .* ce;

            if usingConfiguredLoadCases
                dc = dc + localLoadSensitivityForCase( ...
                    Ui, ue, loadCases(icase), dMdxScale, ME, yDown, ...
                    harmonicPhi, harmonicOmegas, PhiLegacy, omegaLegacy, edofMat, ...
                    nodalProjectionCache, semiHarmonicBaseVec);
            end
        end
        clear K M;

        dv = ones(nEl, 1);
        % Passive elements are excluded from the volume constraint and OC update.
        dv(pasS) = 0;  dv(pasV) = 0;
        dc(pasS) = 0;  dc(pasV) = 0;

        % Sensitivity filtering
        if ft == 0
            dc = (Hf * (x .* dc)) ./ Hs ./ max(0.001, x);
        elseif ft == 1
            dc = Hf * (dc ./ Hs);
            dv = Hf * (dv ./ Hs);
        end
        % Re-zero passive indices after filtering (filter spreads values to neighbours).
        dc(pasS) = 0;  dc(pasV) = 0;
        dv(pasS) = 0;  dv(pasV) = 0;

        % Design update: OC or MMA (selected by runCfg.optimizer).
        xold = x;
        if strcmp(optimizerType, 'OC')
            [x, g] = oc(nelx, nely, x, volfrac, dc, dv, g, move);
            % Restore passive densities (OC may have perturbed them slightly).
            x(pasS) = 1;  x(pasV) = 0;
        else  % MMA
            % Only active elements are passed to mmasub.  Passive elements are
            % excluded: they have x fixed at 0 or 1, so xmin==xmax==x would make
            % mmasub's xl1 = xval-low = 0 → 1/0 → NaN in subsolv.
            % Global [0,1] bounds are used so mmasub's asyinit=0.01 initialises
            % asymptotes at ±0.01 of the full range — the standard working scale.
            n_act_cur = numel(act);
            xmin_act  = zeros(n_act_cur, 1);
            xmax_act  = ones(n_act_cur, 1);
            % Volume constraint: f_1 = sum(xPhys)/nEl - volfrac <= 0.
            % Gradient w.r.t. active x only; dv already chain-rule-filtered.
            fval_mma = sum(xPhys) / nEl - volfrac;
            mmaConstraintPre = fval_mma;
            dfdx_act = dv(act)' / nEl;   % 1 x n_act_cur
            [xnew_act, ~, ~, ~, ~, ~, ~, ~, ~, mma_low, mma_upp] = ...
                mmasub(1, n_act_cur, loop, x(act), xmin_act, xmax_act, ...
                       mma_xold1, mma_xold2, ...
                       obj, dc(act), fval_mma, dfdx_act, ...
                       mma_low, mma_upp, ...
                       1, zeros(1,1), 1e3*ones(1,1), ones(1,1));
            % Enforce user-specified move limit.  mmasub uses asymptotes that can
            % widen to 0.2*(xmax-xmin)=0.2; without this clip the actual per-element
            % step can far exceed the requested move_limit (e.g. 0.18 >> 0.05).
            xnew_act = max(max(zeros(n_act_cur,1), x(act) - move), ...
                           min(min(ones(n_act_cur,1), x(act) + move), xnew_act));
            mma_xold2 = mma_xold1;
            mma_xold1 = x(act);
            x(act) = xnew_act;
            x(pasS) = 1;  x(pasV) = 0;
        end

        % Filter design variables
        xPhys = localPhysicalFieldFromDesign(x, ft, Hf, Hs, pasS, pasV);

        if strcmp(optimizerType, 'MMA')
            mmaConstraintPost = mean(xPhys) - volfrac;
            if mmaConstraintPost > 1e-10
                [x, xPhys, mmaConstraintPost, mmaProjected] = localProjectMmaToVolume( ...
                    x, xold, act, move, ft, Hf, Hs, pasS, pasV, volfrac);
            end
        end

        % Current volume and change
        vol    = mean(xPhys);
        change = max(abs(x - xold));

        for icase = 1:nCases
            msg = sprintf('  load_case[%d] \"%s\": ||F||=%.3e', ...
                icase, loadCases(icase).name, caseDiag(icase).normF);
            if ~isempty(caseDiag(icase).closestNodeIds)
                msg = sprintf('%s, closest_node ids=%s', msg, mat2str(caseDiag(icase).closestNodeIds));
            end
            if ~isempty(caseDiag(icase).harmonicModes)
                msg = sprintf('%s, harmonic=%s', msg, ...
                    localFormatHarmonicDiag(caseDiag(icase).harmonicModes, caseDiag(icase).harmonicOmegas));
            end
            if ~isempty(caseDiag(icase).semiHarmonicModes)
                msg = sprintf('%s, semi_harmonic=%s', msg, ...
                    localFormatHarmonicDiag(caseDiag(icase).semiHarmonicModes, caseDiag(icase).semiHarmonicOmegas));
            end
            fprintf('%s\n', msg);
        end

        if visualiseLive
            if usingConfiguredLoadCases && ~isempty(harmonicOmegas) && isfinite(harmonicOmegas(1))
                omegaTitle = harmonicOmegas(1);
            elseif ~usingConfiguredLoadCases
                omegaTitle = omegaLegacy;
            else
                omegaTitle = NaN;
            end
            plotTopology( ...
                xPhys, nelx, nely, ...
                formatTopologyTitle(approachName, volfrac, omegaTitle), ...
                true, 'regular', false);
        end

        if strcmp(optimizerType, 'MMA')
            if mmaProjected
                fprintf('  [MMA volume projection] applied, residual=%.3e\n', mmaConstraintPost);
            end
            fprintf(['it.: %4d , obj(sum_i u_i^T K u_i): %.3f Vol.: %.3f, ' ...
                     'mma_pre: %.3e, mma_post: %.3e, ch.: %.3f\n'], ...
                    loop, obj, vol, mmaConstraintPre, mmaConstraintPost, change);
        else
            fprintf('it.: %4d , obj(sum_i u_i^T K u_i): %.3f Vol.: %.3f, ch.: %.3f\n', ...
                    loop, obj, vol, change);
        end
    end
    loop_time = toc(loop_tic);
    tIter = loop_time / max(loop, 1);
    nIter = loop;
    if saveFrqIterations
        info.freq_iter_omega = info.freq_iter_omega(1:loop,:);
    end
    info.last_F = F;
    info.last_U = U;
    info.last_obj = obj;
    info.last_obj_cases = objCases;
    info.load_case_names = {loadCases.name};
    info.last_vol = mean(xPhys);
    if debugReturnDc
        info.last_dc = dc;
        info.last_dv = dv;
    end

    % ------------------------------------------------------------------
    % Post-analysis: first circular frequency for final topology
    % ------------------------------------------------------------------
    sK_final = reshape(KE(:) * (Emin + xPhys'.^penal * (Emax - Emin)), [], 1);
    K_final  = sparse(iK, jK, sK_final, ndof, ndof);
    K_final  = (K_final + K_final') / 2;

    rhoPhys_final = rho_min + xPhys.^pmass * (rho0 - rho_min);
    sM_final = reshape(ME(:) * rhoPhys_final', [], 1);
    M_final  = sparse(iK, jK, sM_final, ndof, ndof);
    M_final  = (M_final + M_final') / 2;

    Kf_final = K_final(free, free);
    Mf_final = M_final(free, free);
    clear K_final sK_final M_final sM_final rhoPhys_final;
    nReq = min(3, max(1, size(Kf_final, 1) - 1));
    [~, Lam_final] = eigs(Kf_final, Mf_final, nReq, 'smallestabs');
    lam_vals = sort(real(diag(Lam_final)), 'ascend');
    lam_vals = lam_vals(lam_vals > 0);
    fHz = NaN(3,1);
    if ~isempty(lam_vals)
        nOk = min(3, numel(lam_vals));
        fHz(1:nOk) = sqrt(lam_vals(1:nOk)) / (2*pi);
    end
    lam1_final = NaN;
    omega1_final = NaN;
    f1_final = NaN;
    if ~isempty(lam_vals)
        lam1_final = lam_vals(1);
        omega1_final = sqrt(lam1_final);
        f1_final = fHz(1);
    end
    fprintf('[Final Eigen] lambda1=%.6e, omega1=%.6e rad/s, f1=%.6e Hz\n', ...
            lam1_final, omega1_final, f1_final);
    plotTopology( ...
        xPhys, nelx, nely, ...
        formatTopologyTitle(approachName, volfrac, omega1_final), ...
        visualiseLive, visualizationQuality, true);

    xOut = xPhys(:);
end

function [loadCases, usingConfiguredLoadCases, maxHarmonicMode, modeUpdateAfter, maxSemiHarmonicMode] = localResolveLoadCases(runCfg, nodeX, nodeY, nodeIds)
usingConfiguredLoadCases = false;
if isstruct(runCfg) && isfield(runCfg, 'load_cases')
    if isempty(runCfg.load_cases)
        error('topopt_freq:EmptyLoadCases', ...
            'runCfg.load_cases is present but empty. Provide at least one case or omit the field.');
    end
    if exist('validateLoadCases', 'file') == 2
        loadCases = validateLoadCases(runCfg.load_cases, 'domain.load_cases');
    else
        error('topopt_freq:MissingLoadCaseValidator', ...
            'validateLoadCases.m is required to parse runCfg.load_cases.');
    end
    usingConfiguredLoadCases = true;
else
    legacyLoad = struct('type', 'harmonic', 'factor', [], ...
        'location', [], 'force', [], 'mode', 1);
    loadCases = struct('name', 'legacy_harmonic_fixed_mode', ...
        'factor', 1.0, 'loads', legacyLoad);
end

loadCases = loadCases(:);
maxHarmonicMode = 0;
maxSemiHarmonicMode = 0;
modeUpdateAfterRaw = [];  % grows dynamically; Inf sentinel, replaced by min(ua) per mode
for icase = 1:numel(loadCases)
    for j = 1:numel(loadCases(icase).loads)
        ld = loadCases(icase).loads(j);
        switch ld.type
            case 'closest_node'
                loc = ld.location(:)';
                dist2 = (nodeX - loc(1)).^2 + (nodeY - loc(2)).^2;
                minD2 = min(dist2);
                % Deterministic tie-break: smallest node id.
                nodeId = min(nodeIds(dist2 == minD2));
                loadCases(icase).loads(j).node_id = nodeId;
            case 'harmonic'
                modeK = ld.mode;
                maxHarmonicMode = max(maxHarmonicMode, modeK);
                ua = 1;  % default: recompute every iteration
                if isfield(ld, 'update_after') && ~isempty(ld.update_after)
                    ua = ld.update_after;
                end
                % Grow sentinel array (Inf) if needed, then take minimum across
                % all references to the same mode (most-frequent update wins).
                if numel(modeUpdateAfterRaw) < modeK
                    modeUpdateAfterRaw(end+1:modeK) = Inf;
                end
                modeUpdateAfterRaw(modeK) = min(modeUpdateAfterRaw(modeK), ua);
            case 'semi_harmonic'
                maxSemiHarmonicMode = max(maxSemiHarmonicMode, ld.mode);
        end
    end
end
if maxHarmonicMode > 0
    modeUpdateAfter = reshape(modeUpdateAfterRaw(1:maxHarmonicMode), [], 1);
else
    modeUpdateAfter = zeros(0, 1);
end
end

function [harmonicOmegas, harmonicPhi] = localCurrentModesFromSubmatrices(Kf, Mf, free, ndof, maxMode)
harmonicOmegas = NaN(maxMode, 1);
harmonicPhi = zeros(ndof, maxMode);
if maxMode < 1 || isempty(Kf) || isempty(Mf)
    return;
end

nFree = size(Kf, 1);
if nFree < 2
    return;
end
nReq = min(maxMode, nFree - 1);
if nReq < 1
    return;
end

try
    eigOpts = struct('disp', 0, 'maxit', 800, 'tol', 1e-8);
    [V, D] = eigs(Kf, Mf, nReq, 'smallestabs', eigOpts);
catch
    [V, D] = eigs(Kf, Mf, nReq, 'smallestabs');
end

lamVals = real(diag(D));
[lamVals, order] = sort(lamVals, 'ascend');
V = V(:, order);
valid = isfinite(lamVals) & lamVals > 0;
lamVals = lamVals(valid);
V = V(:, valid);

nOk = min(maxMode, numel(lamVals));
for k = 1:nOk
    phi = V(:, k);
    mn = real(phi' * (Mf * phi));
    if mn > 0
        phi = phi / sqrt(mn);
    end
    harmonicOmegas(k) = sqrt(lamVals(k));
    phiGlobal = zeros(ndof, 1);
    phiGlobal(free) = phi;
    harmonicPhi(:, k) = phiGlobal;
end
end

function [F, caseDiag, harmonicNormRef, semiDebugAssembly] = localBuildLoadMatrix( ...
    loadCases, ndof, M, yDown, harmonicPhi, harmonicOmegas, PhiLegacy, omegaLegacy, ...
    harmonicNormRef, harmonicNormalize, rhoNodal, semiHarmonicBaseVec, semiHarmonicOmega0, ...
    captureSemiDebug)
if nargin < 14 || isempty(captureSemiDebug)
    captureSemiDebug = false;
end
nCases = numel(loadCases);
F = zeros(ndof, nCases);
semiDebugAssembly = struct('caseIdx', {}, 'loadIdx', {}, 'mode', {}, ...
    'loadCaseFactor', {}, 'loadFactor', {}, 'coeff', {}, 'fSemi', {});
caseDiag = repmat(struct( ...
    'normF', 0.0, ...
    'closestNodeIds', [], ...
    'harmonicModes', [], ...
    'harmonicOmegas', [], ...
    'semiHarmonicModes', [], ...
    'semiHarmonicOmegas', []), nCases, 1);

if ~isempty(rhoNodal)
    rhoNodal = reshape(rhoNodal, [], 1);
    rhoDof = zeros(ndof, 1);
    rhoDof(1:2:end) = rhoNodal;
    rhoDof(2:2:end) = rhoNodal;
else
    rhoDof = [];
end

for icase = 1:nCases
    Fi = zeros(ndof, 1);
    closestNodeIds = [];
    harmonicModes = [];
    harmonicUsed = [];
    semiModes = [];
    semiUsed = [];

    loads = loadCases(icase).loads;
    for j = 1:numel(loads)
        ld = loads(j);
        ldFactor = localLoadFactor(ld);
        switch ld.type
            case 'self_weight'
                Fi = Fi + ldFactor * (M * yDown);

            case 'closest_node'
                nodeId = ld.node_id;
                Fi(2*nodeId - 1) = Fi(2*nodeId - 1) + ldFactor * ld.force(1);
                Fi(2*nodeId) = Fi(2*nodeId) + ldFactor * ld.force(2);
                closestNodeIds(end+1) = nodeId; %#ok<AGROW>

            case 'harmonic'
                modeK = ld.mode;
                if modeK <= numel(harmonicOmegas) && isfinite(harmonicOmegas(modeK))
                    omegaK = harmonicOmegas(modeK);
                    phiK = harmonicPhi(:, modeK);
                elseif isfinite(omegaLegacy)
                    omegaK = omegaLegacy;
                    phiK = PhiLegacy;
                else
                    error('topopt_freq:HarmonicModeUnavailable', ...
                        'Unable to evaluate harmonic mode %d for load_cases case \"%s\".', ...
                        modeK, loadCases(icase).name);
                end
                fH = (omegaK^2) * (M * phiK);
                if harmonicNormalize && modeK <= numel(harmonicNormRef)
                    nRaw = norm(fH);
                    if (~isfinite(harmonicNormRef(modeK)) || harmonicNormRef(modeK) <= 0) && nRaw > 0
                        harmonicNormRef(modeK) = nRaw;
                    end
                    nRef = harmonicNormRef(modeK);
                    if isfinite(nRef) && nRef > 0 && nRaw > 0
                        fH = fH * (nRef / nRaw);
                    end
                end
                Fi = Fi + ldFactor * fH;
                harmonicModes(end+1) = modeK; %#ok<AGROW>
                harmonicUsed(end+1) = omegaK; %#ok<AGROW>

            case 'semi_harmonic'
                modeK = ld.mode;
                if isempty(rhoDof)
                    error('topopt_freq:MissingSemiHarmonicNodalDensity', ...
                        'rho_nodal projection is required for semi_harmonic load assembly.');
                end
                if modeK > size(semiHarmonicBaseVec, 2) || ...
                        modeK > numel(semiHarmonicOmega0) || ...
                        ~isfinite(semiHarmonicOmega0(modeK))
                    error('topopt_freq:SemiHarmonicModeUnavailable', ...
                        'Unable to evaluate semi_harmonic mode %d for load_cases case \"%s\".', ...
                        modeK, loadCases(icase).name);
                end
                fSemi = rhoDof .* semiHarmonicBaseVec(:, modeK);
                if captureSemiDebug
                    dbgIdx = numel(semiDebugAssembly) + 1;
                    semiDebugAssembly(dbgIdx).caseIdx = icase;
                    semiDebugAssembly(dbgIdx).loadIdx = j;
                    semiDebugAssembly(dbgIdx).mode = modeK;
                    semiDebugAssembly(dbgIdx).loadCaseFactor = loadCases(icase).factor;
                    semiDebugAssembly(dbgIdx).loadFactor = ldFactor;
                    semiDebugAssembly(dbgIdx).coeff = loadCases(icase).factor * ldFactor;
                    semiDebugAssembly(dbgIdx).fSemi = fSemi;
                end
                Fi = Fi + ldFactor * fSemi;
                semiModes(end+1) = modeK; %#ok<AGROW>
                semiUsed(end+1) = semiHarmonicOmega0(modeK); %#ok<AGROW>
        end
    end

    Fi = loadCases(icase).factor * Fi;
    F(:, icase) = Fi;

    caseDiag(icase).normF = norm(Fi);
    caseDiag(icase).closestNodeIds = unique(closestNodeIds, 'stable');
    caseDiag(icase).harmonicModes = harmonicModes;
    caseDiag(icase).harmonicOmegas = harmonicUsed;
    caseDiag(icase).semiHarmonicModes = semiModes;
    caseDiag(icase).semiHarmonicOmegas = semiUsed;
end
end

function dcLoad = localLoadSensitivityForCase( ...
    Ui, ue, loadCase, dMdxScale, ME, yDown, harmonicPhi, harmonicOmegas, ...
    PhiLegacy, omegaLegacy, edofMat, nodalProjectionCache, semiHarmonicBaseVec)
% Load-dependent sensitivity contributions (dF/drho_e term in dJ/drho_e).
%
% Option B for harmonic loads:
%   dF_h/dx ≈ ld.factor * omega_k^2 * (dM/dx * Phi_k)
% while d(omega_k)/dx and d(Phi_k)/dx are intentionally ignored.
dcLoad = zeros(size(dMdxScale));
loads = loadCase.loads;

for j = 1:numel(loads)
    ld = loads(j);
    ldFactor = localLoadFactor(ld);
    coeff = 0;
    vec = [];
    switch ld.type
        case 'self_weight'
            coeff = loadCase.factor * ldFactor;
            vec = yDown;
        case 'harmonic'
            modeK = ld.mode;
            if modeK <= numel(harmonicOmegas) && isfinite(harmonicOmegas(modeK))
                omegaK = harmonicOmegas(modeK);
                phiK = harmonicPhi(:, modeK);
            elseif isfinite(omegaLegacy)
                omegaK = omegaLegacy;
                phiK = PhiLegacy;
            else
                error('topopt_freq:HarmonicModeUnavailable', ...
                    'Unable to evaluate harmonic mode %d for load_cases case \"%s\".', ...
                    modeK, loadCase.name);
            end
            coeff = loadCase.factor * ldFactor;
            vec = (omegaK^2) * phiK;
        case 'semi_harmonic'
            % Frozen-load approximation: treat the semi_harmonic inertial
            % load direction as fixed w.r.t. x.  Differentiating through
            % rhoNodal(x) adds a positive term (+2 U^T dF/dx > 0) that
            % counteracts the stiffness sensitivity and drives the optimizer
            % to remove material from high-mode-amplitude regions — the
            % opposite of the frequency-maximization objective.
            % The semi_harmonic method is a frozen-load heuristic; load
            % sensitivity is intentionally zeroed here, analogous to the
            % frozen-eigenpair approximation used for harmonic loads.
            continue;
        otherwise
            % closest_node has no rho dependence.
    end

    if coeff == 0 || isempty(vec)
        continue;
    end
    vec_e = vec(edofMat);
    uMeVec = sum((ue * ME) .* vec_e, 2);
    dcLoad = dcLoad + 2 * coeff * dMdxScale .* uMeVec;
end
end

function fac = localLoadFactor(ld)
fac = 1.0;
if isstruct(ld) && isfield(ld, 'factor') && ~isempty(ld.factor)
    if ~isnumeric(ld.factor) || ~isscalar(ld.factor) || ~isfinite(ld.factor)
        error('topopt_freq:InvalidLoadFactor', ...
            'Load factor must be a finite numeric scalar when provided.');
    end
    fac = double(ld.factor);
end
end

function msg = localFormatHarmonicDiag(modes, omegas)
if isempty(modes)
    msg = '[]';
    return;
end
parts = cell(numel(modes), 1);
for i = 1:numel(modes)
    if i <= numel(omegas) && isfinite(omegas(i))
        parts{i} = sprintf('mode%d(omega=%.3e)', modes(i), omegas(i));
    else
        parts{i} = sprintf('mode%d(omega=NaN)', modes(i));
    end
end
msg = ['[' strjoin(parts, ', ') ']'];
end

function omegas = localFirstNOmegasFromSubmatrices(Kf, Mf, nModes)
    omegas = NaN(1, nModes);
    if nModes < 1 || isempty(Kf) || isempty(Mf)
        return;
    end
    nReq = min(nModes, max(1, size(Kf, 1) - 1));
    if nReq < 1
        return;
    end
    try
        eigOpts = struct('disp', 0, 'maxit', 800, 'tol', 1e-8);
        Lam = eigs(Kf, Mf, nReq, 'smallestabs', eigOpts);
        lamVals = sort(real(diag(Lam)), 'ascend');
        lamVals = lamVals(lamVals > 0);
        nOk = min(nModes, numel(lamVals));
        if nOk > 0
            omegas(1:nOk) = sqrt(lamVals(1:nOk));
        end
    catch
        omegas(:) = NaN;
    end
end

function xBase = localBuildSemiHarmonicBaseline(kind, xPhys, pasS, pasV, nEl)
switch lower(strtrim(kind))
    case 'solid'
        xBase = ones(nEl, 1);
        if ~isempty(pasV), xBase(pasV) = 0; end
        if ~isempty(pasS), xBase(pasS) = 1; end
    case 'initial'
        % Legacy behavior used the initial xPhys field at cache time.
        xBase = reshape(xPhys, [], 1);
    otherwise
        error('topopt_freq:InvalidSemiHarmonicBaseline', ...
            'Unknown semi_harmonic baseline "%s".', kind);
end
xBase = reshape(double(xBase), [], 1);
if numel(xBase) ~= nEl
    error('topopt_freq:InvalidSemiHarmonicBaselineSize', ...
        'semi_harmonic baseline must have %d elements (got %d).', nEl, numel(xBase));
end
end

function rhoSourceVec = localSemiHarmonicRhoSourceVector(source, x, xPhys)
switch lower(strtrim(source))
    case 'x'
        rhoSourceVec = x;
    case 'xphys'
        rhoSourceVec = xPhys;
    otherwise
        error('topopt_freq:InvalidSemiHarmonicRhoSource', ...
            'Unknown semi_harmonic rho source "%s".', source);
end
rhoSourceVec = reshape(double(rhoSourceVec), [], 1);
end

function localDebugSemiHarmonicIteration( ...
    loop, loadCases, semiDebugAssembly, ndof, free, Kf, ...
    rhoSourceVec, rhoNodal, nelx, nely, nodalProjectionCache, ...
    semiHarmonicBaselineInfo, semiHarmonicM0, semiHarmonicPhi0, semiHarmonicOmega0, ...
    semiHarmonicBaseVec, act)

fprintf('[semi_harmonic debug] it=%d\n', loop);
fprintf(['  baseline=%s, rho_source=%s, xBase[min/mean/max]=[%.4f, %.4f, %.4f], ' ...
         'passives(solid=%d, void=%d)\n'], ...
    semiHarmonicBaselineInfo.kind, semiHarmonicBaselineInfo.rhoSource, ...
    semiHarmonicBaselineInfo.xBaseMin, semiHarmonicBaselineInfo.xBaseMean, ...
    semiHarmonicBaselineInfo.xBaseMax, ...
    semiHarmonicBaselineInfo.passiveSolidCount, semiHarmonicBaselineInfo.passiveVoidCount);

if ~isempty(semiHarmonicOmega0)
    M0f = semiHarmonicM0(free, free);
    for k = 1:numel(semiHarmonicOmega0)
        if ~isfinite(semiHarmonicOmega0(k))
            continue;
        end
        phiK = semiHarmonicPhi0(:, k);
        mPhi = semiHarmonicM0 * phiK;
        baseK = semiHarmonicOmega0(k) * mPhi;
        mn = real(phiK(free)' * (M0f * phiK(free)));
        fprintf(['  mode %d: omega0=%.6e rad/s, ||phi||=%.6e, ||M0*phi||=%.6e, ' ...
                 '||omega0*(M0*phi)||=%.6e, phi^T M0 phi=%.6e\n'], ...
            k, semiHarmonicOmega0(k), norm(phiK), norm(mPhi), norm(baseK), mn);
    end
end

localDebugCheckSemiFormula( ...
    loop, rhoNodal, semiDebugAssembly, ...
    semiHarmonicM0, semiHarmonicPhi0, semiHarmonicOmega0, semiHarmonicBaseVec);
localDebugCheckSemiProjectionConsistency(loop, rhoSourceVec, rhoNodal, nodalProjectionCache);
fprintf('  FD check uses fixed K and perturbs rho_source=%s only.\n', semiHarmonicBaselineInfo.rhoSource);
localDebugCheckSemiFiniteDiff( ...
    loop, loadCases, ndof, free, Kf, rhoSourceVec, rhoNodal, ...
    nelx, nely, nodalProjectionCache, semiHarmonicBaseVec, semiHarmonicOmega0, act);
end

function localDebugCheckSemiFormula( ...
    loop, rhoNodal, semiDebugAssembly, semiHarmonicM0, semiHarmonicPhi0, semiHarmonicOmega0, semiHarmonicBaseVec)
if isempty(semiDebugAssembly)
    fprintf('  formula check: no semi_harmonic loads active in this iteration.\n');
    return;
end

rhoNodal = reshape(rhoNodal, [], 1);
ndof = 2 * numel(rhoNodal);
rhoDof = zeros(ndof, 1);
rhoDof(1:2:end) = rhoNodal;
rhoDof(2:2:end) = rhoNodal;

for q = 1:numel(semiDebugAssembly)
    entry = semiDebugAssembly(q);
    modeK = entry.mode;
    baseRef = semiHarmonicOmega0(modeK) * (semiHarmonicM0 * semiHarmonicPhi0(:, modeK));
    fRef = rhoDof .* baseRef;
    fAct = entry.fSemi;
    relErr = norm(fRef - fAct) / max(norm(fRef), eps);
    maxAbs = max(abs(fRef - fAct));
    baseRelErr = norm(baseRef - semiHarmonicBaseVec(:, modeK)) / max(norm(baseRef), eps);
    fprintf(['  formula check it=%d case[%d] load[%d] mode=%d: relerr=%.3e, maxabs=%.3e, ' ...
             'basevec_relerr=%.3e\n'], ...
        loop, entry.caseIdx, entry.loadIdx, modeK, relErr, maxAbs, baseRelErr);
end
end

function localDebugCheckSemiProjectionConsistency(loop, rhoSourceVec, rhoNodal, nodalProjectionCache)
rhoSourceVec = reshape(rhoSourceVec, [], 1);
nNodes = nodalProjectionCache.nNodes;
idx = nodalProjectionCache.elemNodes(:);
vals = repmat(rhoSourceVec, 4, 1);
sumVals = accumarray(idx, vals, [nNodes, 1], @sum, 0);
counts = accumarray(idx, 1, [nNodes, 1], @sum, 0);
rhoNodalRef = sumVals ./ max(counts, 1);

rhoNodal = reshape(rhoNodal, [], 1);
maxDiffAccum = max(abs(rhoNodal - rhoNodalRef));
rhoNodalFromPavg = nodalProjectionCache.Pavg * rhoSourceVec;
maxDiffPavg = max(abs(rhoNodal - rhoNodalFromPavg));

fprintf(['  projection check it=%d: max|rho_proj-rho_accum|=%.3e, ' ...
         'max|rho_proj-Pavg*x|=%.3e\n'], ...
    loop, maxDiffAccum, maxDiffPavg);

nEl = numel(rhoSourceVec);
s = rng; cleanupRng = onCleanup(@() rng(s)); %#ok<NASGU>
rng(9103 + loop, 'twister');
sampleElems = sort(randperm(nEl, min(3, nEl)));
for e = sampleElems(:)'
    nodes = nodalProjectionCache.elemNodes(e, :);
    expected = 1 ./ counts(nodes);
    actual = full(nodalProjectionCache.Pavg(nodes, e));
    maxAbs = max(abs(actual(:) - expected(:)));
    fprintf('  Pavg weight check e=%d: max|actual-expected|=%.3e\n', e, maxAbs);
end
end

function localDebugCheckSemiFiniteDiff( ...
    loop, loadCases, ndof, free, Kf, rhoSourceVec, rhoNodal, ...
    nelx, nely, nodalProjectionCache, semiHarmonicBaseVec, semiHarmonicOmega0, act)

Fsemi0 = localBuildSemiOnlyLoadMatrix( ...
    loadCases, ndof, rhoNodal, semiHarmonicBaseVec, semiHarmonicOmega0);
if ~any(Fsemi0(:))
    fprintf('  FD check it=%d: skipped (semi_harmonic load norm is zero).\n', loop);
    return;
end

U0 = zeros(ndof, size(Fsemi0, 2));
U0(free, :) = Kf \ Fsemi0(free, :);
J0 = sum(sum(U0(free, :) .* Fsemi0(free, :)));
dcSemi = localSemiHarmonicSensitivityFromU(U0, loadCases, nodalProjectionCache, semiHarmonicBaseVec);

candidates = act(:);
if isempty(candidates)
    candidates = (1:numel(rhoSourceVec))';
end
nPick = min(3, numel(candidates));
if nPick < 1
    fprintf('  FD check it=%d: skipped (no candidate elements).\n', loop);
    return;
end

s = rng; cleanupRng = onCleanup(@() rng(s)); %#ok<NASGU>
rng(1731 + loop, 'twister');
sel = candidates(randperm(numel(candidates), nPick));
delta = 1e-6;
for ee = 1:numel(sel)
    e = sel(ee);
    rho0 = rhoSourceVec(e);
    rho1 = min(1.0, rho0 + delta);
    if rho1 == rho0
        rho1 = max(0.0, rho0 - delta);
    end
    dxe = rho1 - rho0;
    if dxe == 0
        fprintf('  FD check e=%d: skipped (bounded at both ends).\n', e);
        continue;
    end

    rhoPert = rhoSourceVec;
    rhoPert(e) = rho1;
    rhoNodalPert = projectQ4ElementDensityToNodes(rhoPert, nelx, nely, nodalProjectionCache);
    Fsemi1 = localBuildSemiOnlyLoadMatrix( ...
        loadCases, ndof, rhoNodalPert, semiHarmonicBaseVec, semiHarmonicOmega0);
    U1 = zeros(ndof, size(Fsemi1, 2));
    U1(free, :) = Kf \ Fsemi1(free, :);
    J1 = sum(sum(U1(free, :) .* Fsemi1(free, :)));

    fd = (J1 - J0) / dxe;
    ana = dcSemi(e);
    relErr = abs(fd - ana) / max(abs(fd), eps);
    fprintf('  FD check e=%d: analytic=%.6e, fd=%.6e, relerr=%.3e\n', e, ana, fd, relErr);
end
end

function Fsemi = localBuildSemiOnlyLoadMatrix(loadCases, ndof, rhoNodal, semiHarmonicBaseVec, semiHarmonicOmega0)
nCases = numel(loadCases);
Fsemi = zeros(ndof, nCases);
if isempty(rhoNodal)
    return;
end

rhoNodal = reshape(rhoNodal, [], 1);
rhoDof = zeros(ndof, 1);
rhoDof(1:2:end) = rhoNodal;
rhoDof(2:2:end) = rhoNodal;

for icase = 1:nCases
    Fi = zeros(ndof, 1);
    loads = loadCases(icase).loads;
    for j = 1:numel(loads)
        ld = loads(j);
        if ~strcmp(ld.type, 'semi_harmonic')
            continue;
        end
        modeK = ld.mode;
        if modeK > size(semiHarmonicBaseVec, 2) || ...
                modeK > numel(semiHarmonicOmega0) || ...
                ~isfinite(semiHarmonicOmega0(modeK))
            error('topopt_freq:SemiHarmonicModeUnavailable', ...
                'Unable to evaluate semi_harmonic mode %d for load_cases case \"%s\".', ...
                modeK, loadCases(icase).name);
        end
        coeff = loadCases(icase).factor * localLoadFactor(ld);
        Fi = Fi + coeff * (rhoDof .* semiHarmonicBaseVec(:, modeK));
    end
    Fsemi(:, icase) = Fi;
end
end

function dcSemi = localSemiHarmonicSensitivityFromU(U, loadCases, nodalProjectionCache, semiHarmonicBaseVec)
nEl = size(nodalProjectionCache.Pavg, 2);
dcSemi = zeros(nEl, 1);
for icase = 1:numel(loadCases)
    Ui = U(:, icase);
    loads = loadCases(icase).loads;
    for j = 1:numel(loads)
        ld = loads(j);
        if ~strcmp(ld.type, 'semi_harmonic')
            continue;
        end
        coeff = loadCases(icase).factor * localLoadFactor(ld);
        if coeff == 0
            continue;
        end
        modeK = ld.mode;
        baseVec = semiHarmonicBaseVec(:, modeK);
        nodeTerm = Ui(1:2:end) .* baseVec(1:2:end) + Ui(2:2:end) .* baseVec(2:2:end);
        dcSemi = dcSemi + 2 * coeff * (nodalProjectionCache.Pavg' * nodeTerm);
    end
end
end


% ======================================================================
% Element stiffness matrix (Q4, plane stress) for rectangular element hx x hy
% ======================================================================
function KE = lk(hx, hy, nu)
    E  = 1.0;
    D  = (E / (1 - nu^2)) * [1, nu, 0; nu, 1, 0; 0, 0, 0.5*(1-nu)];

    invJ = [2/hx, 0; 0, 2/hy];
    detJ = 0.25 * hx * hy;
    gp   = 1 / sqrt(3);
    gauss_pts = [-gp, gp];

    KE_ccw = zeros(8, 8);
    for xi = gauss_pts
        for eta = gauss_pts
            dN_dxi  = 0.25 * [-(1-eta),  (1-eta),  (1+eta), -(1+eta)];
            dN_deta = 0.25 * [-(1-xi),  -(1+xi),   (1+xi),   (1-xi)];
            dN_xy = invJ * [dN_dxi; dN_deta];
            dN_dx = dN_xy(1, :);
            dN_dy = dN_xy(2, :);

            B = zeros(3, 8);
            B(1, 1:2:end) = dN_dx;
            B(2, 2:2:end) = dN_dy;
            B(3, 1:2:end) = dN_dy;
            B(3, 2:2:end) = dN_dx;

            KE_ccw = KE_ccw + (B' * D * B) * detJ;
        end
    end

    % Standard Q4 node order (LL, LR, UR, UL), matching edofMat above.
    KE = KE_ccw;
end


% ======================================================================
% Element consistent mass matrix (Q4, 2 dof/node) for rectangular hx x hy
% ======================================================================
function ME = lm(hx, hy)
    area = hx * hy;
    Ms_ccw = (area / 36) * [4, 2, 1, 2;
                             2, 4, 2, 1;
                             1, 2, 4, 2;
                             2, 1, 2, 4];
    ME_ccw = kron(Ms_ccw, eye(2));

    % Standard Q4 node order (LL, LR, UR, UL), matching edofMat above.
    ME = ME_ccw;
end


% ======================================================================
% Optimality criteria update
% ======================================================================
function [xnew, gt] = oc(nelx, nely, x, volfrac, dc, dv, g, move)
    l1   = 0;
    l2   = 1e9;
    xnew = zeros(nelx*nely, 1);
    eps_val = 1e-30;

    % Enforce OC assumptions: dv > 0 and dc <= 0
    dv_safe = max(dv, 1e-12);
    dc_safe = min(dc, -1e-12);

    for iter = 1:200
        lmid = 0.5 * (l1 + l2);
        denom = max(lmid, 1e-30);

        B = -dc_safe ./ dv_safe / denom;
        B = max(B, 1e-30);

        x_candidate = x .* sqrt(B);
        xnew = max(0, max(x - move, min(1, min(x + move, x_candidate))));

        gt = g + sum(dv_safe .* (xnew - x));

        if gt > 0
            l1 = lmid;
        else
            l2 = lmid;
        end

        if (l2 - l1) / max(l1 + l2, eps_val) < 1e-3
            break;
        end
    end
end

function xPhys = localPhysicalFieldFromDesign(x, ft, Hf, Hs, pasS, pasV)
if ft == 0
    xPhys = x;
elseif ft == 1
    xPhys = (Hf * x) ./ Hs;
else
    error('topopt_freq:UnsupportedFilterType', 'Unsupported filter type ft=%d.', ft);
end
if ~isempty(pasS), xPhys(pasS) = 1; end
if ~isempty(pasV), xPhys(pasV) = 0; end
end

function [xProj, xPhysProj, residual, projected] = localProjectMmaToVolume( ...
    x, xold, act, move, ft, Hf, Hs, pasS, pasV, volfrac)
% Enforce physical volume feasibility after MMA update via monotone bisection.
% Keeps active variables within [xold-move, xold+move] and [0,1].
xProj = x;
xPhysProj = localPhysicalFieldFromDesign(xProj, ft, Hf, Hs, pasS, pasV);
residual = mean(xPhysProj) - volfrac;
projected = false;
if residual <= 0 || isempty(act)
    return;
end

lb = max(0, xold(act) - move);
ub = min(1, xold(act) + move);
xActBase = min(ub, max(lb, xProj(act)));

% If even maximal downward move is infeasible, return best feasible-by-move point.
xLow = xProj;
xLow(act) = lb;
xLow(pasS) = 1;
xLow(pasV) = 0;
xPhysLow = localPhysicalFieldFromDesign(xLow, ft, Hf, Hs, pasS, pasV);
resLow = mean(xPhysLow) - volfrac;
if resLow > 0
    xProj = xLow;
    xPhysProj = xPhysLow;
    residual = resLow;
    projected = true;
    return;
end

tauLo = 0;
tauHi = max(max(xActBase - lb), 1e-12);
xBest = xLow;
xPhysBest = xPhysLow;
resBest = resLow;
for it = 1:40
    tau = 0.5 * (tauLo + tauHi);
    xTryAct = min(ub, max(lb, xActBase - tau));
    xTry = xProj;
    xTry(act) = xTryAct;
    xTry(pasS) = 1;
    xTry(pasV) = 0;
    xPhysTry = localPhysicalFieldFromDesign(xTry, ft, Hf, Hs, pasS, pasV);
    resTry = mean(xPhysTry) - volfrac;
    if resTry > 0
        tauLo = tau;
    else
        tauHi = tau;
        xBest = xTry;
        xPhysBest = xPhysTry;
        resBest = resTry;
    end
end

xProj = xBest;
xPhysProj = xPhysBest;
residual = resBest;
projected = true;
end

function fixed = localBuildFixedDofs(supportType, nelx, nely)
j_mid = floor(nely/2);
nL = j_mid;
nR = nelx*(nely+1) + j_mid;
leftNodes = (0:nely)';
rightNodes = nelx*(nely+1) + (0:nely)';

switch upper(string(supportType))
    case "SS"
        fixed = [2*nL+1, 2*nL+2, 2*nR+1, 2*nR+2];
    case "CS"
        fixed = [2*leftNodes+1; 2*leftNodes+2; 2*nR+1; 2*nR+2];
    case "CC"
        fixed = [2*leftNodes+1; 2*leftNodes+2; 2*rightNodes+1; 2*rightNodes+2];
    case {"CF","CANTILEVER"}
        fixed = [2*leftNodes+1; 2*leftNodes+2];
    case "NONE"
        % No standard hinge/clamp — all fixed DOFs come from extraFixedDofs.
        fixed = [];
    otherwise
        error('Unsupported supportType "%s" for ourApproach.', string(supportType));
end
fixed = unique(fixed(:))';
end

function v = localOpt(s, name, defaultVal)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = defaultVal;
end
end

function localEnsurePlotHelpersOnPath()
if exist('plotTopology', 'file') == 2 && exist('formatTopologyTitle', 'file') == 2
    return;
end
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
toolsDir = fullfile(repoRoot, 'tools');
if exist(toolsDir, 'dir') == 7
    addpath(toolsDir);
end
end

function tf = localParseVisualiseLive(value, defaultValue)
if nargin < 2
    defaultValue = true;
end
if isempty(value)
    tf = defaultValue;
    return;
end
if islogical(value) && isscalar(value)
    tf = value;
    return;
end
if isnumeric(value) && isscalar(value)
    tf = value ~= 0;
    return;
end
if isstring(value) && isscalar(value)
    value = char(value);
end
if ischar(value)
    key = lower(strtrim(value));
    if any(strcmp(key, {'yes','y','true','1','on'}))
        tf = true;
        return;
    end
    if any(strcmp(key, {'no','n','false','0','off'}))
        tf = false;
        return;
    end
end
error('topopt_freq:InvalidVisualiseLive', ...
    'visualise_live must be yes/no (case-insensitive) or boolean-like.');
end

function quality = localParseVisualizationQuality(value)
if isstring(value) && isscalar(value)
    value = char(value);
end
if ischar(value)
    key = lower(strtrim(value));
    if isempty(key)
        quality = 'regular';
        return;
    end
    if any(strcmp(key, {'regular', 'smooth'}))
        quality = key;
        return;
    end
end
error('topopt_freq:InvalidVisualizationQuality', ...
    'visualization_quality must be "regular" or "smooth".');
end

function baseline = localParseSemiHarmonicBaseline(value)
if isstring(value) && isscalar(value)
    value = char(value);
end
if ischar(value)
    key = lower(strtrim(value));
    switch key
        case 'solid'
            baseline = 'solid';
            return;
        case 'initial'
            baseline = 'initial';
            return;
    end
end
error('topopt_freq:InvalidSemiHarmonicBaseline', ...
    'semi_harmonic_baseline must be "solid" or "initial".');
end

function src = localParseSemiHarmonicRhoSource(value)
if isstring(value) && isscalar(value)
    value = char(value);
end
if ischar(value)
    key = lower(strtrim(value));
    switch key
        case 'x'
            src = 'x';
            return;
        case {'xphys', 'x_phys'}
            src = 'xPhys';
            return;
    end
end
error('topopt_freq:InvalidSemiHarmonicRhoSource', ...
    'semi_harmonic_rho_source must be "x" or "xPhys".');
end

function name = localApproachName(runCfg, defaultName)
if isstruct(runCfg) && isfield(runCfg, 'approach_name') && ~isempty(runCfg.approach_name)
    name = char(string(runCfg.approach_name));
else
    name = defaultName;
end
end
