% A TOPOLOGY OPTIMIZATION CODE FOR FREQUENCY MAXIMIZATION
% Rewritten from the Python version (Aage & Johansen, 2013, modified)
%
% (1) Compute (omega1, Phi1) once on the DESIGN DOMAIN (free DOFs)
% (2) Use harmonic-type load: F(x) = omega1^2 * M(x) * Phi1
% (3) During TO, update F only through M(x) (SIMP mass); Phi1, omega1 stay fixed

function [xOut, fHz, tIter, nIter] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, varargin)
    xOut = [];
    fHz = NaN(3, 1);
    tIter = NaN;
    nIter = NaN;

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

    fprintf('Compliance with harmonic-type inertial load (fixed mode)\n');
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
    if isfield(runCfg, 'visualise_live') && ~isempty(runCfg.visualise_live)
        visualiseLive = localParseVisualiseLive(runCfg.visualise_live, true);
    else
        visualiseLive = true;
    end

    ndof = 2*(nelx+1)*(nely+1);

    % Allocate design variables
    x     = volfrac * ones(nelx*nely, 1);
    xold  = x;
    xPhys = x;
    g     = 0;

    % FE: element stiffness & mass matrices
    KE = lk(hx, hy, nu);
    ME = lm(hx, hy);

    % Build edofMat (0-based DOF numbering converted to 1-based)
    edofMat = zeros(nelx*nely, 8);
    for elx = 0:nelx-1
        for ely = 0:nely-1
            el  = ely + elx*nely + 1;  % 1-based element index
            n1  = (nely+1)*elx + ely;  % 0-based node
            n2  = (nely+1)*(elx+1) + ely;
            edofMat(el,:) = [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, ...
                             2*n2,   2*n2+1, 2*n1,   2*n1+1] + 1; % +1 for 1-based
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

    % Convert support type to constrained DOFs (1-based indexing).
    fixed = localBuildFixedDofs(supportType, nelx, nely);
    alldofs = 1:ndof;
    free = setdiff(alldofs, fixed);

    f = zeros(ndof, 1);
    u = zeros(ndof, 1);

    % ------------------------------------------------------------------
    % (1) EIGENANALYSIS on design domain (free DOFs) once, using initial xPhys
    % ------------------------------------------------------------------
    sK0 = reshape(KE(:) * (Emin + xPhys'.^penal * (Emax - Emin)), [], 1);
    K0  = sparse(iK, jK, sK0, ndof, ndof);
    K0  = (K0 + K0') / 2;

    rhoPhys0 = rho_min + xPhys.^pmass * (rho0 - rho_min);
    sM0 = reshape(ME(:) * rhoPhys0', [], 1);
    M0  = sparse(iK, jK, sM0, ndof, ndof);
    M0  = (M0 + M0') / 2;

    K0f = K0(free, free);
    M0f = M0(free, free);

    % Smallest eigenpair: K phi = lambda M phi (shift-invert near 0)
    [Phi_free, Lam] = eigs(K0f, M0f, 2, 'smallestabs');
    lam_vals = diag(Lam);
    [lam1, idx] = min(lam_vals);
    omega1 = sqrt(max(lam1, 0));

    phi1_free = Phi_free(:, idx);
    mn = phi1_free' * (M0f * phi1_free);
    if mn > 0
        phi1_free = phi1_free / sqrt(mn);
    end

    Phi1 = zeros(ndof, 1);
    Phi1(free) = phi1_free;

    fprintf('[Eigen] lambda1=%.6e, omega1=%.6e rad/s (computed once, fixed)\n', lam1, omega1);

    % ------------------------------------------------------------------
    % Optimization loop
    % ------------------------------------------------------------------
    loop   = 0;
    change = 1;
    dv = ones(nelx*nely, 1);
    dc = ones(nelx*nely, 1);
    ce = ones(nelx*nely, 1);
    loop_tic = tic;

    while change > convTol && loop < maxIters
        loop = loop + 1;

        % (2,3) Update load using CURRENT M(x):  F = omega1^2 * M(x) * Phi1
        rhoPhys = rho_min + xPhys.^pmass * (rho0 - rho_min);
        sM = reshape(ME(:) * rhoPhys', [], 1);
        M  = sparse(iK, jK, sM, ndof, ndof);
        M  = (M + M') / 2;

        f = (omega1^2) * (M * Phi1);

        % Setup and solve FE problem
        sK = reshape(KE(:) * (Emin + xPhys'.^penal * (Emax - Emin)), [], 1);
        K  = sparse(iK, jK, sK, ndof, ndof);
        K  = (K + K') / 2;

        Kf = K(free, free);

        u(:) = 0;
        u(free) = Kf \ f(free);

        % Objective (compliance): C = f' * u
        obj = f' * u;

        % Element strain energy for stiffness sensitivity
        ue = u(edofMat);  % nelx*nely x 8
        ce = sum((ue * KE) .* ue, 2);

        % Sensitivities: dC/dx = -u^T (dK/dx) u + (df/dx)^T u
        dc_stiff = (-penal * xPhys.^(penal-1) * (Emax - Emin)) .* ce;

        % Load sensitivity term (currently commented out as in Python)
        % phi_e = Phi1(edofMat);
        % uMephi = sum((ue * ME) .* phi_e, 2);
        % dMdx_scale = pmass * (xPhys.^(pmass-1)) * (rho0 - rho_min);
        % dc_load = (omega1^2) * dMdx_scale .* uMephi;

        dc = dc_stiff; % + dc_load;
        dv = ones(nelx*nely, 1);

        % Sensitivity filtering
        if ft == 0
            dc = (Hf * (x .* dc)) ./ Hs ./ max(0.001, x);
        elseif ft == 1
            dc = Hf * (dc ./ Hs);
            dv = Hf * (dv ./ Hs);
        end

        % Optimality criteria
        xold = x;
        [x, g] = oc(nelx, nely, x, volfrac, dc, dv, g, move);

        % Filter design variables
        if ft == 0
            xPhys = x;
        elseif ft == 1
            xPhys = (Hf * x) ./ Hs;
        end

        % Current volume and change
        vol    = (g + volfrac*nelx*nely) / (nelx*nely);
        change = max(abs(x - xold));

        if visualiseLive
            plotTopology( ...
                xPhys, nelx, nely, ...
                formatTopologyTitle(approachName, volfrac, omega1), ...
                true);
        end

        fprintf('it.: %4d , obj(C=f^T u): %.3f Vol.: %.3f, ch.: %.3f\n', ...
                loop, obj, vol, change);
    end
    loop_time = toc(loop_tic);
    tIter = loop_time / max(loop, 1);
    nIter = loop;

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
        visualiseLive);

    xOut = xPhys(:);
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

    % topopt edof order: [UL, UR, LR, LL]
    perm = [7, 8, 5, 6, 3, 4, 1, 2];  % 1-based
    KE = KE_ccw(perm, perm);
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

    perm = [7, 8, 5, 6, 3, 4, 1, 2];  % 1-based
    ME = ME_ccw(perm, perm);
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

function name = localApproachName(runCfg, defaultName)
if isstruct(runCfg) && isfield(runCfg, 'approach_name') && ~isempty(runCfg.approach_name)
    name = char(string(runCfg.approach_name));
else
    name = defaultName;
end
end
