function [x, cFinal, tIter, nIter] = topopt_elastic2D(nelx, nely, volfrac, penal, rmin, ft, L, H, runCfg)
%TOPOPT_ELASTIC2D  Minimum compliance topology optimization (2-D plane stress).
%
%   Based on:
%     Andreassen et al. (2010), "Efficient topology optimization in MATLAB
%     using 88 lines of code", Struct Multidisc Optim 43(1):1-16.
%
%   Inputs
%   ------
%   nelx, nely : element counts in x and y
%   volfrac    : target volume fraction
%   penal      : SIMP penalization exponent
%   rmin       : filter radius in physical units
%   ft         : 0 = sensitivity filter, 1 = density filter
%   L, H       : physical domain dimensions
%   runCfg     : struct with solver settings (see below)
%
%   runCfg fields
%   -------------
%   E0, Emin, nu     — material (E0, Emin are stiffnesses; nu Poisson ratio)
%   rho0             — density (used for self_weight loads only)
%   move, convTol, maxIter — optimisation control
%   optimizer        — 'OC' or 'MMA'
%   extraFixedDofs   — 0-based fixed DOF indices (int array)
%   pasS, pasV       — 0-based passive solid / void element indices
%   loadCases        — cell array of load-case structs
%   visualizeLive    — logical: show live topology plot
%
%   Outputs
%   -------
%   x       : (nelx*nely, 1) final physical density field
%   cFinal  : final compliance value
%   tIter   : average time per iteration [s]
%   nIter   : number of iterations executed

    % ---- Unpack runCfg ----
    E0   = getField(runCfg, 'E0',   1.0);
    Emin = getField(runCfg, 'Emin', 1e-9);
    nu   = getField(runCfg, 'nu',   0.3);
    rho0 = getField(runCfg, 'rho0', 1.0);
    move     = getField(runCfg, 'move',     0.2);
    convTol  = getField(runCfg, 'convTol',  0.01);
    maxIter  = getField(runCfg, 'maxIter',  200);
    optimizer = upper(strtrim(getField(runCfg, 'optimizer', 'OC')));
    visualizeLive = logical(getField(runCfg, 'visualizeLive', false));

    if ~any(strcmp(optimizer, {'OC','MMA'}))
        error('topopt_elastic2D:badOptimizer', ...
            'optimizer must be "OC" or "MMA" (got "%s").', optimizer);
    end

    fprintf('elastic2D — minimum compliance\n');
    fprintf('  mesh %d×%d,  domain %.4g×%.4g,  volfrac %.4g\n', nelx, nely, L, H, volfrac);
    fprintf('  penal %.4g,  rmin %.4g,  filter %s,  optimizer %s\n', ...
        penal, rmin, ternary(ft==0,'sensitivity','density'), optimizer);

    hx = L / nelx;
    hy = H / nely;
    nEl    = nelx * nely;
    nNodes = (nelx + 1) * (nely + 1);
    nDof   = 2 * nNodes;

    % ---- Passive elements ----
    pasS = getIntVec(runCfg, 'pasS') + 1;   % convert to 1-based
    pasV = getIntVec(runCfg, 'pasV') + 1;
    allEl = (1:nEl)';
    act = setdiff(allEl, union(pasS, pasV));

    % ---- Boundary conditions (0-based → 1-based) ----
    extraFixed = getIntVec(runCfg, 'extraFixedDofs') + 1;
    if isempty(extraFixed)
        error('topopt_elastic2D:noFixedDofs', ...
            'No fixed DOFs provided. Specify bc.supports in the JSON config.');
    end
    fixedDofs = unique(extraFixed(:));
    freeDofs  = setdiff((1:nDof)', fixedDofs);

    % ---- Element stiffness matrix ----
    KE = keElastic(hx, hy, nu);

    % ---- Element connectivity (LL→LR→UR→UL, 1-based DOFs) ----
    edofMat = zeros(nEl, 8, 'int32');
    for elx = 0:nelx-1
        for ely = 0:nely-1
            el = ely + elx*nely + 1;       % 1-based element index
            n1 = (nely+1)*elx + ely + 1;   % LL node (1-based)
            n2 = (nely+1)*(elx+1) + ely + 1; % LR
            n3 = n2 + 1;                    % UR
            n4 = n1 + 1;                    % UL
            edofMat(el,:) = int32([2*n1-1, 2*n1, 2*n2-1, 2*n2, ...
                                   2*n3-1, 2*n3, 2*n4-1, 2*n4]);
        end
    end
    iK = reshape(kron(edofMat, ones(8,1,'int32'))', 64*nEl, 1);
    jK = reshape(kron(edofMat, ones(1,8,'int32'))', 64*nEl, 1);

    % ---- Filter ----
    rminx = max(1, ceil(rmin / hx));
    rminy = max(1, ceil(rmin / hy));
    nFilt = nEl * (2*(rminx-1)+1) * (2*(rminy-1)+1);
    iH = zeros(nFilt, 1);
    jH = zeros(nFilt, 1);
    sH = zeros(nFilt, 1);
    cc = 0;
    for i = 1:nelx
        for j = 1:nely
            row = (i-1)*nely + j;
            for k = max(i-rminx+1,1):min(i+rminx-1,nelx)
                for ll = max(j-rminy+1,1):min(j+rminy-1,nely)
                    col = (k-1)*nely + ll;
                    d = sqrt(((i-k)*hx)^2 + ((j-ll)*hy)^2);
                    cc = cc + 1;
                    iH(cc) = row;  jH(cc) = col;  sH(cc) = max(0, rmin - d);
                end
            end
        end
    end
    Hfilt = sparse(iH(1:cc), jH(1:cc), sH(1:cc), nEl, nEl);
    Hs    = full(sum(Hfilt, 2));

    % ---- Node coordinates ----
    nodeIds = (0:nNodes-1)';
    nodeX   = floor(nodeIds / (nely+1)) * hx;
    nodeY   = mod(nodeIds, nely+1) * hy;

    % ---- Load vector assembly ----
    [F, caseFactors] = assembleLoads(runCfg, nDof, nelx, nely, hx, hy, rho0, nodeX, nodeY);
    nCases = size(F, 2);
    fprintf('  load cases: %d\n', nCases);

    % ---- MMA setup ----
    if strcmp(optimizer, 'MMA')
        % Ensure tools/Matlab is on path for mmasub.
        thisDir = fileparts(mfilename('fullpath'));
        repoRoot = fileparts(fileparts(fileparts(thisDir)));
        addpath(fullfile(repoRoot, 'tools', 'Matlab'));
        nAct  = numel(act);
        xminM = zeros(nAct, 1);
        xmaxM = ones(nAct, 1);
        xold1 = repmat(volfrac, nAct, 1);
        xold2 = repmat(volfrac, nAct, 1);
        lowM  = xminM;
        uppM  = xmaxM;
    end

    % ---- Design variables ----
    x = repmat(volfrac, nEl, 1);
    x(pasS) = 1.0;
    x(pasV) = 0.0;
    if ~isempty(act) && (~isempty(pasS) || ~isempty(pasV))
        actTarget = (volfrac*nEl - numel(pasS)) / numel(act);
        x(act) = max(0, min(1, actTarget));
    end

    xPhys = physicalField(x, ft, Hfilt, Hs, pasS, pasV);

    % ---- Visualization ----
    hFig = [];
    hImg = [];
    if visualizeLive
        hFig = figure('Color','white','Name','elastic2D topology');
        hImg = imagesc(1 - reshape(xPhys, nely, nelx));
        colormap(gray); axis equal off; caxis([0 1]);
        set(gca, 'YDir', 'normal');
        drawnow;
    end

    % ---- Main loop ----
    loop   = 0;
    change = 1.0;
    cFinal = Inf;
    tTotal = 0.0;
    U      = zeros(nDof, nCases);

    fprintf('  %5s  %14s  %7s  %8s\n', 'It', 'Compliance', 'Vol', 'Ch');

    while change > convTol && loop < maxIter
        t0 = tic;
        loop = loop + 1;

        % FE analysis
        Efield = Emin + xPhys(:)'.^penal * (E0 - Emin);
        sK = reshape(KE(:) * Efield, 64*nEl, 1);
        K  = sparse(double(iK), double(jK), sK, nDof, nDof);
        K  = (K + K') / 2;
        Kf = K(freeDofs, freeDofs);
        for ci = 1:nCases
            U(freeDofs, ci) = Kf \ F(freeDofs, ci);
        end

        % Compliance and sensitivity
        c  = 0;
        ce = zeros(nEl, 1);
        for ci = 1:nCases
            ue   = U(edofMat, ci);           % (nEl, 8)
            ce_k = sum((ue * KE) .* ue, 2);
            c    = c  + caseFactors(ci) * (Efield(:) .* ce_k);
            ce   = ce + caseFactors(ci) * ce_k;
        end
        c = sum(c);
        cFinal = c;

        dc = -penal * (E0 - Emin) * xPhys(:).^(penal-1) .* ce;
        dv = ones(nEl, 1);

        % Sensitivity / density filter
        if ft == 0
            dc_f = (Hfilt * (x .* dc)) ./ Hs ./ max(1e-3, x);
            dv_f = dv;
        else
            dc_f = (Hfilt * dc) ./ Hs;
            dv_f = (Hfilt * dv) ./ Hs;
        end

        volTarget = volfrac * nEl - numel(pasS);

        if strcmp(optimizer, 'OC')
            dcAct = dc_f(act);
            dvAct = dv_f(act);
            xAct  = x(act);
            l1 = 0;  l2 = 1e9;
            while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3
                lmid    = 0.5 * (l1 + l2);
                xnewAct = max(0, max(xAct - move, ...
                              min(1, min(xAct + move, ...
                              xAct .* sqrt(-dcAct ./ dvAct / lmid)))));
                if ft == 1
                    xTrial = x;
                    xTrial(act)  = xnewAct;
                    xTrial(pasS) = 1.0;
                    xTrial(pasV) = 0.0;
                    xpTrial = full(Hfilt * xTrial) ./ Hs;
                    xpTrial(pasS) = 1.0;
                    xpTrial(pasV) = 0.0;
                    volCurr = sum(xpTrial(act));
                else
                    volCurr = sum(xnewAct);
                end
                if volCurr > volTarget
                    l1 = lmid;
                else
                    l2 = lmid;
                end
            end
            xOld   = x(act);
            x(act) = xnewAct;

        else  % MMA
            nAct  = numel(act);
            f0val = c;
            df0dx = dc_f(act);
            fval  = (sum(xPhys(act)) - volTarget) / nEl;
            dfdx  = dv_f(act)' / nEl;
            xCol  = x(act);
            [xnewMMA, ~, ~, ~, ~, ~, ~, ~, ~, lowM, uppM] = mmasub( ...
                1, nAct, loop, ...
                xCol, xminM, xmaxM, ...
                xold1, xold2, ...
                f0val, df0dx, fval, dfdx, ...
                lowM, uppM, 1, zeros(1,1), 1e3*ones(1,1), ones(1,1));
            xold2   = xold1;
            xold1   = x(act);
            xOld    = x(act);
            x(act)  = xnewMMA;
        end

        change   = max(abs(x(act) - xOld));
        x(pasS)  = 1.0;
        x(pasV)  = 0.0;
        xPhys    = physicalField(x, ft, Hfilt, Hs, pasS, pasV);
        tTotal   = tTotal + toc(t0);

        fprintf('  %5d  %14.6g  %7.4f  %8.4f\n', loop, c, mean(xPhys), change);

        if visualizeLive && ~isempty(hImg)
            set(hImg, 'CData', 1 - reshape(xPhys, nely, nelx));
            drawnow;
        end
    end

    tIter = tTotal / max(loop, 1);
    nIter = loop;
    fprintf('  Done: %d iters,  compliance = %.6g,  t/iter = %.3f s\n', loop, cFinal, tIter);

    if visualizeLive && ~isempty(hFig)
        close(hFig);
    end
    x = xPhys(:);
end


% ===========================================================================
% Local functions
% ===========================================================================

function KE = keElastic(hx, hy, nu)
% Q4 plane-stress element stiffness matrix (E=1, LL→LR→UR→UL node order).
% Computed by 2×2 Gauss quadrature on a rectangle of size hx × hy.
    D    = (1/(1-nu^2)) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];
    invJ = diag([2/hx, 2/hy]);
    detJ = 0.25 * hx * hy;
    gp   = 1/sqrt(3);
    KE   = zeros(8, 8);
    for xi = [-gp, gp]
        for eta = [-gp, gp]
            dNdxi  = 0.25 * [-(1-eta),  (1-eta),  (1+eta), -(1+eta)];
            dNdeta = 0.25 * [-(1-xi),  -(1+xi),   (1+xi),  (1-xi)];
            dNxy = invJ * [dNdxi; dNdeta];
            dNdx = dNxy(1,:);  dNdy = dNxy(2,:);
            B = zeros(3, 8);
            B(1,1:2:end) = dNdx;
            B(2,2:2:end) = dNdy;
            B(3,1:2:end) = dNdy;
            B(3,2:2:end) = dNdx;
            KE = KE + (B' * D * B) * detJ;
        end
    end
end


function xp = physicalField(x, ft, Hfilt, Hs, pasS, pasV)
% Compute physical density from design variable x.
    if ft == 0
        xp = x;
    else
        xp = full(Hfilt * x) ./ Hs;
    end
    if ~isempty(pasS), xp(pasS) = 1.0; end
    if ~isempty(pasV), xp(pasV) = 0.0; end
end


function [F, caseFactors] = assembleLoads(runCfg, nDof, nelx, nely, hx, hy, rho0, nodeX, nodeY)
% Assemble load matrix F (nDof × nCases) from load_cases in runCfg.
    if ~isfield(runCfg, 'loadCases') || isempty(runCfg.loadCases)
        error('topopt_elastic2D:noLoads', ...
            'No load cases defined. Specify domain.load_cases in the JSON config.');
    end
    loadCases = runCfg.loadCases;
    if ~iscell(loadCases), loadCases = {loadCases}; end
    nCases = numel(loadCases);
    F = zeros(nDof, nCases);
    caseFactors = ones(nCases, 1);

    % Precompute element corner nodes for self_weight (1-based).
    elIdx  = (0:nelx*nely-1)';
    elxArr = floor(elIdx / nely);
    elyArr = mod(elIdx, nely);
    swN1 = (nely+1)*elxArr + elyArr + 1;           % LL
    swN2 = (nely+1)*(elxArr+1) + elyArr + 1;       % LR
    swN3 = swN2 + 1;                                % UR
    swN4 = swN1 + 1;                                % UL

    for ci = 1:nCases
        lc = loadCases{ci};
        caseFactors(ci) = getField(lc, 'factor', 1.0);
        loads = lc.loads;
        if ~iscell(loads), loads = {loads}; end
        for li = 1:numel(loads)
            ld  = loads{li};
            lt  = lower(strtrim(ld.type));
            ldf = getField(ld, 'factor', 1.0);
            switch lt
                case 'closest_node'
                    loc  = ld.location(:);
                    dist2 = (nodeX - loc(1)).^2 + (nodeY - loc(2)).^2;
                    minD2 = min(dist2);
                    n = find(dist2 == minD2, 1, 'first');   % 1-based node
                    fx = ld.force(1);  fy = ld.force(2);
                    F(2*n-1, ci) = F(2*n-1, ci) + ldf * fx;
                    F(2*n,   ci) = F(2*n,   ci) + ldf * fy;
                case 'self_weight'
                    % Lumped body force: rho0 * factor * hx*hy / 4 per corner in -y.
                    ew = ldf * rho0 * hx * hy / 4.0;
                    for nc = [swN1, swN2, swN3, swN4]'
                        idx = 2 * nc;      % uy DOFs (1-based)
                        for k = 1:numel(idx)
                            F(idx(k), ci) = F(idx(k), ci) - ew;
                        end
                    end
                otherwise
                    error('topopt_elastic2D:badLoadType', ...
                        'Load type "%s" not supported by elastic2D. Supported: closest_node, self_weight.', lt);
            end
        end
    end
end


function v = getField(s, name, default)
% Return s.(name) if it exists, otherwise default.
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    elseif nargin >= 3
        v = default;
    else
        error('topopt_elastic2D:missingField', 'Required field "%s" is missing.', name);
    end
end


function v = getIntVec(s, name)
% Return 0-based integer vector from struct field (empty if absent).
    if isfield(s, name) && ~isempty(s.(name))
        v = int64(s.(name)(:));
    else
        v = int64([]);
    end
end


function s = ternary(cond, a, b)
% Inline ternary for string selection.
    if cond, s = a; else, s = b; end
end
