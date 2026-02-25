function [omega_best, xPhys_best, diagnostics] = topFreqOptimization_MMA(cfg, varargin)

% ==============================================================
% Frequency maximization via MMA (BOUND formulation)
% Maximize min_{j=1..J} lambda_j using variable Eb = E/lambda_ref.
%
% Task parameters (geometry, mesh, materials, continuation schedule, etc.)
% are supplied in struct `cfg` so each example script can tweak them in one
% place (see Huang cases for the pattern). Legacy positional arguments are
% still accepted for backwards compatibility.
% ==============================================================
%
% Output:
%   omega_best  [rad/s]
%   xPhys_best  physical density field
% ==============================================================

% Handle legacy positional usage: (L,H,nelx,nely,volfrac,penal,rmin,maxiter,supportType,J,opts)
legacyOpts = struct();
if nargin >= 1 && ~isstruct(cfg)
    [cfg, legacyOpts] = legacyArgsToCfg(cfg, varargin{:});
    varargin = {};  % legacy opts already captured
end

if nargin < 1 || isempty(cfg), cfg = struct(); end
if isempty(varargin)
    opts = struct();
else
    if isstruct(varargin{1})
        opts = varargin{1};
    else
        opts = struct(varargin{:});
    end
end
opts = mergeStructs(opts, legacyOpts);  % legacy opts win if provided

cfg  = applyDefaults(cfg);
opts = applyDefaultOpts(opts, cfg);
diagnostics = struct();
localEnsurePlotHelpersOnPath();
plotLive = localParseVisualiseLive(opts.visualise_live, true);
visualizationQuality = localParseVisualizationQuality(opts.visualization_quality);
approachName = localApproachName(opts, 'Olhoff');
saveFrqIterations = localParseVisualiseLive(opts.save_frq_iterations, false);

% Shorthand locals (keeps the algorithm body intact)
L          = cfg.L;
H          = cfg.H;
nelx       = cfg.nelx;
nely       = cfg.nely;
volfrac    = cfg.volfrac;
penal      = cfg.penal;
rmin       = cfg.rmin;
maxiter    = cfg.maxiter;
supportType= cfg.supportType;
J          = cfg.J;

% --- material -------------------------------------------------
E0      = cfg.E0;
Emin    = cfg.Emin;
rho0    = cfg.rho0;
rho_min = cfg.rho_min;
nu      = cfg.nu;
t       = cfg.t;

%% --- mesh -----------------------------------------------------
dx = L/nelx; dy = H/nely;
nEl  = nelx*nely;

nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
nDof    = 2*(nelx+1)*(nely+1);

fixed = buildSupports(supportType,nodeNrs);
if isfield(cfg, 'extraFixedDofs') && ~isempty(cfg.extraFixedDofs)
    fixed = unique([fixed(:); cfg.extraFixedDofs(:)]);
end
free  = setdiff(1:nDof,fixed);

% Passive element sets (forced density = 1 or 0).
pasS = []; pasV = [];
if isfield(cfg, 'pasS') && ~isempty(cfg.pasS), pasS = cfg.pasS(:); end
if isfield(cfg, 'pasV') && ~isempty(cfg.pasV), pasV = cfg.pasV(:); end

%% --- element matrices ----------------------------------------
[Ke0, M0] = q4_rect_KeM_planeStress(E0,nu,rho0,t,dx,dy);
Ke = Ke0/E0;
Me = M0/rho0;

%% --- filter ---------------------------------------------------
% rmin is in PHYSICAL units (e.g., rmin = 2*dx means 2 element widths)
% Convert to element units for filter kernel construction
rmin_elem = rmin / dx;  % filter radius in element units
if rmin_elem < 1.2
    warning('rmin=%.4f gives only %.2f elements - checkerboard likely. Use rmin >= 1.5*dx.', rmin, rmin_elem);
    rmin_elem = max(rmin_elem, 1.5);  % minimum 1.5 elements to avoid checkerboard
end
fprintf('Filter: rmin=%.4f (physical) = %.2f elements\n', rmin, rmin_elem);

[dyf,dxf] = meshgrid(-ceil(rmin_elem)+1:ceil(rmin_elem)-1);
h  = max(0, rmin_elem - sqrt(dxf.^2 + dyf.^2));
Hs = imfilter(ones(nely,nelx), h, 'symmetric');

fwd  = @(x) imfilter(x,h,'symmetric')./Hs;
bwd  = @(g) imfilter(g./Hs,h,'symmetric');

%% --- projection ----------------------------------------------
eta = cfg.eta;
betaMax = cfg.betaMax;
beta_prev = -inf;  % track changes for continuation
% TIME-BASED beta schedule: advance every beta_interval iterations
% Start at beta=1, go more aggressively to higher values
beta_list = cfg.beta_schedule(:).';  % ensure row vector
beta_interval = cfg.beta_interval;   % advance beta every N iterations
beta_idx = cfg.beta_start_idx;
Nsafe = cfg.beta_safe_iters;         % safe iterations after beta jump
move_safe = cfg.move_safe;           % safe move limit during Nsafe iters
safe_counter = 0;
gray_tol = cfg.gray_tol;             % relaxed threshold for grayness check
reduce_counter = 0;   % move reduction when eigs fails
move_reduce = cfg.move_reduce;       % increased from 0.01
prev_V_low = [];
Npolish = cfg.Npolish;               % minimum iterations at max beta

% Grayness penalty parameters (adaptive with beta)
% gray_penalty_weight increases with beta to push toward discrete values
% Higher penalty helps drive discreteness for frequency optimization
gray_penalty_base = cfg.gray_penalty_base;  % base penalty weight (scaled by beta/betaMax)

%% --- assembly indices ----------------------------------------
cVec = 2*nodeNrs(1:nely,1:nelx)+1;
cVec = cVec(:);
cMat = [cVec cVec+1 cVec+2*nely+2 cVec+2*nely+3 ...
        cVec+2*nely cVec+2*nely+1 cVec-2 cVec-1];

[Il,Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)',[],1);
jK = reshape(cMat(:,Jl)',[],1);
Ke_l = Ke(sub2ind([8,8],Il,Jl));
Me_l = Me(sub2ind([8,8],Il,Jl));

% helper: evaluate eigenvalues for a given physical density field
evalModes = @(xPhys,numModes) evalEigen(xPhys,numModes,penal,E0,Emin,rho0,rho_min, ...
                                       Ke_l,Me_l,iK,jK,nDof,free);

%% --- MMA setup -----------------------------------------------
lambda_ref = cfg.lambda_ref;         % scaling reference
n = nEl + 1;
m = J + 2;                % J eig constraints + two-sided volume constraint

xmin = [1e-3*ones(nEl,1); cfg.Eb_min];
xmax = [ones(nEl,1);  cfg.Eb_max];     % Eb upper bound raised for target ω ≈ 456 rad/s

xval  = [volfrac*ones(nEl,1); cfg.Eb0];
% Pin passive element design variables at their forced densities.
xmin(pasS) = 1;  xmax(pasS) = 1;  xval(pasS) = 1;
xmin(pasV) = 0;  xmax(pasV) = 0;  xval(pasV) = 0;
xold1 = xval;
xold2 = xval;
low   = xmin;
upp   = xmax;

a0 = 1;
a  = zeros(m,1);
c  = [100*ones(J,1); 10; 10];   % stronger penalty on eigen constraints
d  = 1e-3*ones(m,1);

omega_best = -inf;
xPhys_best = xval(1:nEl);
move = cfg.move;
move_hist_len = cfg.move_hist_len;  % for convergence checks
omega_hist = [];
dx_hist = [];
if saveFrqIterations
    freqIterOmega = NaN(maxiter, 3);
end

%% --- diagnostic: uniform design --------------------------------
if opts.doDiagnostic
    beta0 = 1; % initial projection slope
    x0 = volfrac*ones(nEl,1);
    xT0 = fwd(reshape(x0,nely,nelx));
    [xPhys0,~] = heavisideProjection(xT0,beta0,eta);
    xPhys0 = xPhys0(:);
    if ~isempty(pasS), xPhys0(pasS) = 1; end
    if ~isempty(pasV), xPhys0(pasV) = 0; end
    [lam0,omega0,freq0] = evalModes(xPhys0,opts.diagModes);
    diagnostics.initial = struct('lam',lam0,'omega',omega0,'freq',freq0);
    fprintf('--- Diagnostic: uniform design, support=%s, vol=%.2f ---\n', upper(string(supportType)), volfrac);
    fmt3 = @(v) sprintf('%8.3f %8.3f %8.3f', v(1:min(3,numel(v))));
    fprintf('  omega [rad/s]: %s\n', fmt3(omega0));
    fprintf('  f     [Hz]   : %s\n', fmt3(freq0));
    fprintf('  (paper reports Z1≈146.1 for CC; compare above)\n');
    if opts.diagnosticOnly
        omega_best = omega0(1);
        xPhys_best = xPhys0;
        diagnostics.final = diagnostics.initial;
        diagnostics.iterations = 0;
        diagnostics.loop_time = 0;
        diagnostics.t_iter = NaN;
        fprintf('Diagnostic-only mode, skipping optimization.\n');
        fprintf('Best omega = %.4f rad/s (%.4f Hz)\n', omega_best, omega_best/(2*pi));
        return;
    end
end

%% =================== OPT LOOP ================================
iter_executed = 0;
loop_tic = tic;
for it = 1:maxiter
    iter_executed = it;

    % --- MMA shape hardening (CRITICAL)
    xval=xval(:); xold1=xold1(:); xold2=xold2(:);
    xmin=xmin(:); xmax=xmax(:);
    low=low(:); upp=upp(:);

    % --- adaptive beta schedule with grayness check ---
    beta_target = beta_list(min(beta_idx, numel(beta_list)));
    beta = min(betaMax, beta_target);

    x = xval(1:nEl);
    Eb = xval(end);

    % --- filter + projection
    xT = fwd(reshape(x,nely,nelx));
    [xPhysMat,dH] = heavisideProjection(xT,beta,eta);
    xPhys = xPhysMat(:);
    % Enforce passive element densities (overrides projection).
    if ~isempty(pasS), xPhys(pasS) = 1; end
    if ~isempty(pasV), xPhys(pasV) = 0; end

    % --- assemble K,M
    % Standard SIMP for stiffness, linear for mass (physically correct)
    Ee = Emin + (xPhys.^penal)*(E0-Emin);
    re = rho_min + xPhys*(rho0-rho_min);  % linear mass interpolation

    K = sparse(iK,jK,kron(Ee,Ke_l),nDof,nDof);
    M = sparse(iK,jK,kron(re,Me_l),nDof,nDof);
    K = K+K'-diag(diag(K));
    M = M+M'-diag(diag(M));

    Kf = K(free,free);
    Mf = M(free,free);

    % --- eigensolve: use consistent lowest modes for objective + constraints
    optsSM.tol = 1e-8; optsSM.maxit = 400; optsSM.disp = 0;
    [V_low,D_low,flag_eigs] = eigs(Kf,Mf,J,'SM',optsSM);
    [lam_sorted, idx_sort] = sort(real(diag(D_low)));
    V_low = V_low(:,idx_sort);

    % residual check
    resvec = zeros(J,1);
    for jj=1:J
        vtmp = V_low(:,jj);
        Kv = Kf*vtmp; Mv = Mf*vtmp;
        resvec(jj) = norm(Kv - lam_sorted(jj)*Mv)/(norm(Kv)+eps);
    end
    resmax = max(resvec);

    if flag_eigs ~= 0 || resmax > 1e-3
        optsRetry = optsSM;
        optsRetry.maxit = 1200;
        optsRetry.tol   = 1e-10;
        optsRetry.p     = min(size(Kf,1)-1, max(20, 4*J));
        if ~isempty(prev_V_low)
            optsRetry.v0 = prev_V_low(:,1);
        end
        [V_low,D_low,flag_eigs] = eigs(Kf,Mf,J,'SM',optsRetry);
        [lam_sorted, idx_sort] = sort(real(diag(D_low)));
        V_low = V_low(:,idx_sort);
        for jj=1:J
            vtmp = V_low(:,jj);
            Kv = Kf*vtmp; Mv = Mf*vtmp;
            resvec(jj) = norm(Kv - lam_sorted(jj)*Mv)/(norm(Kv)+eps);
        end
        resmax = max(resvec);
        if flag_eigs ~= 0 || resmax > 1e-3
            reduce_counter = 5;
            fprintf('WARN eigs failed (flag=%d, res=%.2e); skipping update, reducing move for 5 iters\n',...
                flag_eigs,resmax);
            prev_V_low = [];
            beta_prev = beta; % don't trigger beta change next loop
            continue; % skip MMA update this iteration
        end
    end
    prev_V_low = V_low;

    omega_cur = sqrt(lam_sorted(1));
    if saveFrqIterations
        nLog = min(3, numel(lam_sorted));
        freqIterOmega(it,1:nLog) = sqrt(max(lam_sorted(1:nLog), 0));
    end

    % --- handle beta change: re-feasibilize Eb and reset MMA history
    if beta ~= beta_prev
        Eb_feas = min(lam_sorted(1:J))/lambda_ref;
        Eb = min(Eb, Eb_feas - 1e-8);
        Eb = min(xmax(end), max(xmin(end), Eb));
        xval(end) = Eb;
        % restart MMA history/asymptotes to avoid stale steps
        xold1 = xval;
        xold2 = xval;
        low   = xmin;
        upp   = xmax;
        safe_counter = Nsafe;  % enter safe mode for a few iterations
        move = min(move, 0.05); % tighten move after beta jump
    end
    beta_prev = beta;
    if omega_cur > omega_best
        omega_best = omega_cur;
        xPhys_best = xPhys;
    end

    % --- sensitivities
    % SIMP for stiffness: dE/dx = p * x^(p-1) * (E0 - Emin)
    % Linear for mass:    drho/dx = (rho0 - rho_min)
    dlam = zeros(nEl,J);
    for j=1:J
        v = V_low(:,j); v=v/sqrt(v'*(Mf*v));
        phi=zeros(nDof,1); phi(free)=v;
        pe=phi(cMat);
        dlam(:,j)= ...
            penal*(E0-Emin)*(xPhys.^(penal-1)).*sum((pe*Ke).*pe,2) ...
          - lam_sorted(j)*(rho0-rho_min).*sum((pe*Me).*pe,2);
    end

    % --- MMA objective with grayness penalty
    % Grayness measure: g = mean(4*x*(1-x)), ranges 0 (discrete) to 1 (all grey)
    % Derivative: d/dx[4*x*(1-x)] = 4*(1 - 2*x)
    gray_measure = mean(4*xPhys.*(1-xPhys));
    dgray_dxPhys = 4*(1 - 2*xPhys) / nEl;  % derivative w.r.t. each xPhys element

    % Adaptive penalty: increases with beta to allow early exploration
    gray_penalty_weight = gray_penalty_base * (beta / betaMax);

    % Objective: maximize Eb (minimize -Eb) with grayness penalty
    f0 = -Eb + gray_penalty_weight * gray_measure;

    % Gradient w.r.t. design variables (needs chain rule through projection and filter)
    dgray_dxT = reshape(dgray_dxPhys, nely, nelx) .* dH;  % chain through projection
    dgray_dx = bwd(dgray_dxT);  % chain through filter

    df0 = zeros(n,1);
    df0(1:nEl) = gray_penalty_weight * dgray_dx(:);  % grayness penalty gradient
    df0(end) = -1;  % gradient w.r.t. Eb

    % --- constraints
    fval=zeros(m,1); dfdx=zeros(m,n);

    for j=1:J
        scale_j = max(1,Eb);
        fval(j)=(Eb-lam_sorted(j)/lambda_ref)/scale_j;
        g = reshape(-dlam(:,j)/lambda_ref,nely,nelx).*dH;
        bg = bwd(g)/scale_j;
        dfdx(j,1:nEl)=bg(:);
        dfdx(j,end)=1/scale_j;
    end

    % volume upper bound g_up = mean(xPhys)-volfrac <= 0
    g_up = mean(xPhys)-volfrac;
    gv=reshape((1/nEl)*ones(nEl,1),nely,nelx).*dH;
    bgv = bwd(gv);
    fval(J+1) = g_up;
    dfdx(J+1,1:nEl)=bgv(:);
    % volume lower bound g_low = volfrac-mean(xPhys) <= 0
    g_low = volfrac-mean(xPhys);
    fval(J+2) = g_low;
    dfdx(J+2,1:nEl) = -bgv(:);

    % Zero gradients for passive elements so MMA does not try to move them
    % (MMA bounds already enforce this, but zeroing avoids numerical drift).
    if ~isempty(pasS) || ~isempty(pasV)
        allPas = union(pasS(:), pasV(:));
        df0(allPas)      = 0;
        dfdx(:, allPas)  = 0;
    end

    % --- MMA step
    % Ensure all arrays have exactly size n (mmasub can return different sizes)
    ensureSize = @(v,sz) [v(1:min(numel(v),sz)); v(end)*ones(max(0,sz-numel(v)),1)];
    xval  = ensureSize(xval(:), n);
    xold1 = ensureSize(xold1(:), n);
    xold2 = ensureSize(xold2(:), n);
    xmin  = ensureSize(xmin(:), n);
    xmax  = ensureSize(xmax(:), n);
    low   = ensureSize(low(:), n);
    upp   = ensureSize(upp(:), n);
    df0   = ensureSize(df0(:), n);

    [xnew,~,~,~,~,~,~,~,low,upp] = ...
        mmasub(m,n,it,xval,xmin,xmax,xold1,xold2,...
               f0,df0,fval,dfdx,low,upp,a0,a,c,d);

    % Ensure returned arrays have correct size
    xnew = ensureSize(xnew(:), n);
    low  = ensureSize(low(:), n);
    upp  = ensureSize(upp(:), n);

    % damp Eb step to avoid feasibility spikes
    Eb_old = xval(end);
    dEbMax = 0.2*max(1,Eb_old);
    Eb_new = xnew(end);
    Eb_new = min(Eb_old + dEbMax, max(Eb_old - dEbMax, Eb_new));
    Eb_new = min(xmax(end), max(xmin(end), Eb_new));
    xnew(end) = Eb_new;

    % move limit on densities (trust region)
    move_lim = move;
    if safe_counter > 0,     move_lim = min(move_lim, move_safe); end
    if reduce_counter > 0,   move_lim = min(move_lim, move_reduce); end
    xnew(1:nEl) = min(max(xnew(1:nEl), xval(1:nEl)-move_lim), xval(1:nEl)+move_lim);
    % enforce variable bounds
    xnew = max(xmin, min(xmax, xnew));
    if safe_counter > 0, safe_counter = safe_counter - 1; end
    if reduce_counter > 0, reduce_counter = reduce_counter - 1; end

    % guard against objective drop: evaluate trial omega
    x_trial = xnew(1:nEl);
    xT_trial = fwd(reshape(x_trial,nely,nelx));
    [xPhys_trial,~] = heavisideProjection(xT_trial,beta,eta);
    lam_trial = evalModes(xPhys_trial,min(3,J));
    omega_trial = sqrt(lam_trial(1));
    if omega_trial < omega_cur*(1-0.01)
        move = max(move*0.5,0.01);
        xnew(1:nEl) = min(max(xnew(1:nEl), xval(1:nEl)-move), xval(1:nEl)+move);
        xnew = max(xmin, min(xmax, xnew));
    end

    xold2=xold1; xold1=xval; xval=xnew;

    % consider advancing beta if design is discrete enough
    grayness = mean(4*xPhys.*(1 - xPhys));
    frac_gray = mean(xPhys>0.1 & xPhys<0.9);
    % convergence metrics
    if exist('x_prev','var')
        change_x = norm(x - x_prev)/sqrt(nEl);
    else
        change_x = inf;
    end
    if exist('omega_prev','var')
        rel_change_obj = abs(omega_cur-omega_prev)/max(omega_prev,eps);
    else
        rel_change_obj = inf;
    end
    omega_hist = [omega_hist; omega_cur];
    dx_hist    = [dx_hist; change_x];
    if numel(omega_hist) > move_hist_len
        omega_hist(1) = [];
        dx_hist(1) = [];
    end

    % TIME-BASED beta continuation: advance every beta_interval iterations
    % but keep at least Npolish iterations at max beta
    target_beta_idx = min(numel(beta_list), floor(it / beta_interval) + 1);
    if target_beta_idx > beta_idx
        beta_idx = target_beta_idx;
        safe_counter = Nsafe;  % enter safe mode after beta jump
        move = min(move, move_safe);
        fprintf('  -> Advancing beta to %d (iteration %d)\n', beta_list(beta_idx), it);
    end

    % Also advance if design is already discrete
    if grayness < gray_tol && beta_idx < numel(beta_list)
        beta_idx = beta_idx + 1;
        safe_counter = Nsafe;
        move = min(move, move_safe);
        fprintf('  -> Advancing beta to %d (grayness=%.3f < %.3f)\n', ...
            beta_list(beta_idx), grayness, gray_tol);
    end

    if beta >= 16 && grayness > 0.15
        warning('projection continuation not driving discreteness — check sensitivity chain rule (gray=%.3f)',grayness);
    end

    % termination check
    if beta_idx == numel(beta_list)
        polish_left = max(0, Npolish - (it - beta_interval*(numel(beta_list)-1)));
        if rel_change_obj < cfg.conv_tol && change_x < cfg.conv_tol && grayness < 0.05 && polish_left <= 0
            fprintf('Converged: rel dOmega=%.2e, dx=%.2e, gray=%.3f (polish satisfied)\n',rel_change_obj,change_x,grayness);
            break;
        end
    else
        if rel_change_obj < cfg.conv_tol && change_x < cfg.conv_tol && grayness < 0.05
            fprintf('Converged early: rel dOmega=%.2e, dx=%.2e, gray=%.3f\n',rel_change_obj,change_x,grayness);
            break;
        end
    end

    x_prev = x;
    omega_prev = omega_cur;

    max_g_eig = max(fval(1:J));
    % histogram summary for gray diagnostics
    bins_low  = mean(xPhys <= 0.05);
    bins_mid  = mean(xPhys > 0.05 & xPhys < 0.95);
    bins_high = mean(xPhys >= 0.95);

    fprintf('It:%3d  beta:%2d  omega:%.3f  vol:%.3f  gray:%.3f  fracGray:%.3f  bins[0-0.05|mid|0.95-1]=[%.3f %.3f %.3f]  g_up:%+.2e  g_low:%+.2e  maxg:%+.2e  maxg_eig:%+.2e\n',...
        it,beta,omega_cur,mean(xPhys),grayness,frac_gray,bins_low,bins_mid,bins_high,g_up,g_low,max(fval),max_g_eig)

    if plotLive
        omega2_cur = NaN;
        if numel(lam_sorted) >= 2, omega2_cur = sqrt(max(lam_sorted(2), 0)); end
        titleStr = formatTopologyTitle(approachName, volfrac, omega_cur, omega2_cur);
        plotTopology(xPhys, nelx, nely, titleStr, true, 'regular', false);
    end
end
loop_time = toc(loop_tic);
diagnostics.iterations = iter_executed;
diagnostics.loop_time = loop_time;
diagnostics.t_iter = loop_time / max(iter_executed, 1);
if saveFrqIterations
    diagnostics.freq_iter_omega = freqIterOmega(1:iter_executed,:);
end

% --- final diagnostics on best design (use true smallest modes)
[lam_best,omega_vec_best,freq_vec_best] = evalModes(xPhys_best,opts.diagModes);
omega_best = omega_vec_best(1);
diagnostics.final = struct('lam',lam_best,'omega',omega_vec_best,'freq',freq_vec_best);
omega2_best = NaN;
if numel(omega_vec_best) >= 2, omega2_best = omega_vec_best(2); end
plotTopology( ...
    xPhys_best, nelx, nely, ...
    formatTopologyTitle(approachName, volfrac, omega_best, omega2_best), ...
    plotLive, visualizationQuality, true);

fprintf('\nBest design: omega1 = %.4f rad/s (%.4f Hz)\n',omega_best,omega_best/(2*pi))
fprintf('Best design eigenfreqs omega [rad/s]: %s\n', sprintf('%8.3f ', omega_vec_best(1:min(3,end))))
fprintf('Best design eigenfreqs f     [Hz]   : %s\n', sprintf('%8.3f ', freq_vec_best(1:min(3,end))))
if isfield(diagnostics,'initial')
    fprintf('Initial uniform eigenfreqs omega [rad/s]: %s\n', sprintf('%8.3f ', diagnostics.initial.omega(1:min(3,end))))
    fprintf('Initial uniform eigenfreqs f     [Hz]   : %s\n', sprintf('%8.3f ', diagnostics.initial.freq(1:min(3,end))))
end
fprintf('Best design volume = %.4f (target %.4f)\n', mean(xPhys_best), volfrac);
if abs(mean(xPhys_best)-volfrac) > 0.002
    warning('Volume constraint not tight: deviation %.4e', mean(xPhys_best)-volfrac);
end
end


function cfg = applyDefaults(cfg)
    defaults = struct( ...
        'L', 8, ...
        'H', 1, ...
        'nelx', 240, ...
        'nely', 30, ...
        'volfrac', 0.5, ...
        'penal', 3.0, ...
        'rmin', [], ...                      % if empty: 2*L/nelx
        'maxiter', 300, ...
        'supportType', "CC", ...
        'J', 3, ...
        'E0', 1e7, ...
        'Emin', [], ...                      % if empty: max(1e-6*E0,1e-3)
        'rho0', 1.0, ...
        'rho_min', 1e-6, ...
        'nu', 0.3, ...
        't', 1.0, ...
        'eta', 0.5, ...
        'betaMax', 64, ...
        'beta_schedule', [1 2 4 8 16 32 64], ...
        'beta_interval', 40, ...
        'beta_start_idx', 1, ...
        'beta_safe_iters', 5, ...
        'move_safe', 0.05, ...
        'gray_tol', 0.10, ...
        'move_reduce', 0.02, ...
        'Npolish', 40, ...
        'gray_penalty_base', 0.5, ...
        'lambda_ref', 2e4, ...
        'Eb0', 1, ...
        'Eb_min', 0, ...
        'Eb_max', 50, ...
        'move', 0.2, ...
        'move_hist_len', 10);

    cfg = mergeStructs(defaults, cfg);

    if isempty(cfg.rmin)
        cfg.rmin = 2 * cfg.L / cfg.nelx;
    end
    if isempty(cfg.Emin)
        cfg.Emin = max(1e-6 * cfg.E0, 1e-3);
    end
    if isempty(cfg.beta_schedule)
        cfg.beta_schedule = [1 2 4 8 16 32 64];
    end
    if ~isfield(cfg, 'conv_tol') || isempty(cfg.conv_tol)
        cfg.conv_tol = 1e-3;
    end
    cfg.supportType = string(cfg.supportType);
end

function opts = applyDefaultOpts(opts, cfg)
    defaults = struct( ...
        'doDiagnostic', true, ...
        'diagnosticOnly', false, ...
        'diagModes', max(3, cfg.J), ...
        'plotBinary', false, ...
        'visualise_live', true, ...
        'visualization_quality', 'regular', ...
        'save_frq_iterations', false);
    if isfield(cfg, 'visualization_quality') && ~isempty(cfg.visualization_quality)
        defaults.visualization_quality = cfg.visualization_quality;
    end
    opts = mergeStructs(defaults, opts);
end

function out = mergeStructs(base, override)
    out = base;
    if isempty(override)
        return;
    end
    fn = fieldnames(override);
    for k = 1:numel(fn)
        out.(fn{k}) = override.(fn{k});
    end
end

function [cfg, legacyOpts] = legacyArgsToCfg(L,H,nelx,nely,volfrac,penal,rmin,maxiter,supportType,J,varargin)
    if nargin < 10 || isempty(J), J = 2; end
    cfg = struct('L',L,'H',H,'nelx',nelx,'nely',nely,'volfrac',volfrac, ...
                 'penal',penal,'rmin',rmin,'maxiter',maxiter, ...
                 'supportType',supportType,'J',J);
    legacyOpts = struct();
    if ~isempty(varargin)
        if isstruct(varargin{1})
            legacyOpts = varargin{1};
        else
            try
                legacyOpts = struct(varargin{:});
            catch
                warning('Could not parse legacy optional arguments; using default opts.');
                legacyOpts = struct();
            end
        end
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
error('topFreqOptimization_MMA:InvalidVisualiseLive', ...
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
error('topFreqOptimization_MMA:InvalidVisualizationQuality', ...
    'visualization_quality must be "regular" or "smooth".');
end

function name = localApproachName(opts, defaultName)
    if isstruct(opts) && isfield(opts, 'approach_name') && ~isempty(opts.approach_name)
        name = char(string(opts.approach_name));
    else
        name = defaultName;
    end
end

% ---------------- helpers ---------------------------------------
function [xPhys, dH] = heavisideProjection(xTilde, beta, eta)
    denom = tanh(beta*eta) + tanh(beta*(1-eta));
    xPhys = (tanh(beta*eta) + tanh(beta*(xTilde-eta))) / denom;
    dH    = (beta * (1 - tanh(beta*(xTilde-eta)).^2)) / denom;
end

function fixedDOFs = buildSupports(supportType, nodeNrs)
    % Build DOF constraints for different boundary conditions.
    % Paper Figure 2 shows hinges at MID-HEIGHT (neutral axis):
    %   (a) SS: pin supports at mid-height of both edges
    %   (b) CS: left edge clamped, right has pin at mid-height
    %   (c) CC: both edges fully clamped

    nely = size(nodeNrs,1)-1;
    leftNodes  = nodeNrs(:,1);
    rightNodes = nodeNrs(:,end);

    % Corner nodes
    leftBot  = nodeNrs(1,1);
    rightBot = nodeNrs(1,end);

    % MID-HEIGHT nodes (neutral axis) - this is where hinges go per Figure 2
    midIdx = round(nely/2) + 1;  % node index at y = H/2
    leftMid  = nodeNrs(midIdx, 1);
    rightMid = nodeNrs(midIdx, end);

    u = @(n) 2*n - 1;  % x-DOF
    v = @(n) 2*n;      % y-DOF

    s = upper(string(supportType));
    switch s
        case {"CF","CLAMPED-FREE"}
            % Cantilever: left edge clamped, right free
            fixedDOFs = [u(leftNodes(:)); v(leftNodes(:))];

        case {"CC","CLAMPED-CLAMPED"}
            % Both edges fully clamped
            fixedDOFs = [u(leftNodes(:)); v(leftNodes(:)); ...
                        u(rightNodes(:)); v(rightNodes(:))];

        case {"CH","CS","CLAMPED-SIMPLY"}
            % Left edge clamped, right simply-supported at MID-HEIGHT
            % Per Figure 2(b): left hatched, right has pin at neutral axis
            fixedDOFs = [u(leftNodes(:)); v(leftNodes(:)); ...
                        u(rightMid); v(rightMid)];  % pin at mid-height

        case {"HH","SS","SIMPLY-SUPPORTED"}
            % Simply-supported at both ends - pins at MID-HEIGHT (neutral axis)
            % Per Figure 2(a): triangular supports at mid-height of both edges
            % Left: pin (u,v fixed), Right: pin (u,v fixed)
            fixedDOFs = [u(leftMid); v(leftMid); u(rightMid); v(rightMid)];

        case {"SS_CORNER"}
            % Alternative: corner supports (for comparison)
            fixedDOFs = [u(leftBot); v(leftBot); v(rightBot)];

        case "NONE"
            % No standard hinge/clamp — all fixed DOFs come from extraFixedDofs.
            fixedDOFs = [];

        otherwise
            error("Unknown supportType '%s'. Use CC, CS, SS, CF, or NONE.", s);
    end
    fixedDOFs = unique(fixedDOFs(:));
end

function [Ke, Me] = q4_rect_KeM_planeStress(E, nu, rho, t, dx, dy)
    C = E/(1-nu^2) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];

    gp = [-1/sqrt(3),  1/sqrt(3)];
    Ke = zeros(8,8);
    Me = zeros(8,8);

    xiN  = [-1  1  1 -1];
    etaN = [-1 -1  1  1];

    detJ = (dx*dy)/4;
    invJ = [2/dx 0; 0 2/dy];

    for i = 1:2
        for j = 1:2
            xi  = gp(i);
            eta = gp(j);

            N = 0.25 * (1 + xi*xiN) .* (1 + eta*etaN);
            dN_dxi  = 0.25 * xiN  .* (1 + eta*etaN);
            dN_deta = 0.25 * etaN .* (1 + xi*xiN);

            grads = invJ * [dN_dxi; dN_deta];
            dN_dx = grads(1,:);
            dN_dy = grads(2,:);

            B = zeros(3,8);
            for a = 1:4
                B(1,2*a-1) = dN_dx(a);
                B(2,2*a)   = dN_dy(a);
                B(3,2*a-1) = dN_dy(a);
                B(3,2*a)   = dN_dx(a);
            end

            Nmat = zeros(2,8);
            for a = 1:4
                Nmat(1,2*a-1) = N(a);
                Nmat(2,2*a)   = N(a);
            end

            Ke = Ke + (B' * C * B) * (t * detJ);
            Me = Me + (Nmat' * Nmat) * (rho * t * detJ);
        end
    end
end

function [lam,omega,freq] = evalEigen(xPhys, numModes, penal, E0, Emin, rho0, rho_min, ...
                                      Ke_l, Me_l, iK, jK, nDof, free)
    numModes = min(numModes, numel(free));
    Ee = Emin + (xPhys.^penal)*(E0-Emin);
    re = rho_min + xPhys*(rho0-rho_min);  % linear mass interpolation
    K = sparse(iK,jK,kron(Ee,Ke_l),nDof,nDof);
    M = sparse(iK,jK,kron(re,Me_l),nDof,nDof);
    K = K+K'-diag(diag(K));
    M = M+M'-diag(diag(M));
    Kf = K(free,free); Mf = M(free,free);
    [V,D] = eigs(Kf,Mf,numModes,'SM');
    lam = sort(real(diag(D)));
    omega = sqrt(lam);
    freq = omega/(2*pi);
end
