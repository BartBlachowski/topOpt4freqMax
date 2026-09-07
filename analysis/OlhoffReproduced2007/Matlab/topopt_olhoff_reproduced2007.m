function [rho, omega, info] = topopt_olhoff_reproduced2007( ...
    nelx, nely, volfrac, penal, rmin, move, maxit, bcType, runCfg)
%TOPOPT_OLHOFF_REPRODUCED2007  Du--Olhoff 2007 Eq. (22) LP method.
%
%   [RHO,OMEGA,INFO] = TOPOPT_OLHOFF_REPRODUCED2007(NELX,NELY,VOLFRAC,
%   PENAL,RMIN,MOVE,MAXIT,BCTYPE,RUNCFG) applies the clean-room reproduction
%   in Matlab/reproduction2007 to the three problems used by the Yuksel
%   example runners.  BCTYPE is 'simply', 'fixedPinned', or 'cantilever'.
%
%   The cantilever carries a nondesign concentrated mass at the right-edge
%   midpoint.  Its default value matches the Yuksel runner: 20 percent of the
%   permitted material mass, placed on both translational DOFs of that node.
%
%   RUNCFG is optional. Recognized fields are verbose, tol_mult, tol_outer,
%   threads, tip_mass_fraction, mass_interp, filter_mode, rho_min,
%   n_modes_max, optimizer, off_diag, max_inner, tol_inner, and min_inner.
%   The frozen reproduction tree is only read and is installed with its
%   fail-closed path guard for the duration of this call.
%
%   OPTIMIZER selects the Step 3 inner solver of the reproduction:
%
%     'lp'   DEFAULT.  The Du--Olhoff Eq. (22) LP route after Krog & Olhoff
%            (1999): one LINPROG solve per outer iteration.  The vanishing
%            off-diagonal conditions f_sk'drho = 0 are imposed exactly, so
%            OFF_DIAG is necessarily false on this route.
%     'mma'  The paper-literal MMA inner loop on problem (25) -- the labelled
%            baseline of the clean-room study, which NOTES.md 7 records as
%            non-convergent once N >= 2.  Up to MAX_INNER MMA sub-iterates
%            per outer iteration.  OFF_DIAG defaults to true here (the full
%            Eq. (25d) coupling); setting it false routes the Eq. (22)
%            equalities through MMA instead, which the frozen INNERLOOPLP
%            header documents as ill-conditioned (empty feasible interior).
%
%   Both routes call the frozen INNERLOOPLP / INNERLOOP unmodified, so the
%   LP results of this function are unchanged by the added switch.

if nargin < 1 || isempty(nelx), nelx = 320; end
if nargin < 2 || isempty(nely), nely = 40; end
if nargin < 3 || isempty(volfrac), volfrac = 0.5; end
if nargin < 4 || isempty(penal), penal = 3; end
if nargin < 5 || isempty(rmin), rmin = 1.3; end
if nargin < 6 || isempty(move), move = 0.005; end
if nargin < 7 || isempty(maxit), maxit = 400; end
if nargin < 8 || isempty(bcType), bcType = "simply"; end
if nargin < 9 || isempty(runCfg), runCfg = struct(); end
if ~isstruct(runCfg) || ~isscalar(runCfg)
    error('topopt_olhoff_reproduced2007:RunCfg', ...
        'runCfg must be a scalar struct.');
end

validateattributes(nelx, {'numeric'}, {'scalar','integer','positive'});
validateattributes(nely, {'numeric'}, {'scalar','integer','positive'});
validateattributes(volfrac, {'numeric'}, {'scalar','>',0,'<=',1});
validateattributes(penal, {'numeric'}, {'scalar','positive'});
validateattributes(rmin, {'numeric'}, {'scalar','positive'});
validateattributes(move, {'numeric'}, {'scalar','>',0,'<=',1});
validateattributes(maxit, {'numeric'}, {'scalar','integer','positive'});

% Install the frozen implementation without leaking any of its colliding
% function names into the caller's MATLAB session.
thisDir = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(thisDir)));
runnerDir = fullfile(repo, 'Matlab', 'reproduction2007', 'runner');
if exist(runnerDir, 'dir') ~= 7
    error('topopt_olhoff_reproduced2007:MissingReproduction', ...
        'Cannot find Matlab/reproduction2007 below %s.', repo);
end
oldPath = path();
pathCleanup = onCleanup(@() path(oldPath));
addpath(runnerDir);
reproGuard = repro2007_paths(); %#ok<NASGU>
identity = repro2007_assert_identity(false);

[cfg, sourceMeta] = repro2007_config('fig3a_best');
cfg.nelx = double(nelx);
cfg.nely = double(nely);
cfg.volfrac = double(volfrac);
cfg.rho0 = double(volfrac);
cfg.p = double(penal);
cfg.rminEl = double(rmin);
cfg.rminPhys = [];
cfg.move = double(move);
cfg.maxOuter = double(maxit);
cfg.verbose = logical(localOpt(runCfg, 'verbose', true));
cfg.tolMult = double(localOpt(runCfg, 'tol_mult', cfg.tolMult));
cfg.tolOuter = double(localOpt(runCfg, 'tol_outer', cfg.tolOuter));
cfg.threads = double(localOpt(runCfg, 'threads', 1));
cfg.massInterp = char(string(localOpt(runCfg, 'mass_interp', cfg.massInterp)));
cfg.filterMode = char(string(localOpt(runCfg, 'filter_mode', cfg.filterMode)));
cfg.rhomin = double(localOpt(runCfg, 'rho_min', cfg.rhomin));
cfg.Nmax = double(localOpt(runCfg, 'n_modes_max', cfg.Nmax));
[cfg, route] = localOptimizer(cfg, runCfg);
cfg.maxInner = double(localOpt(runCfg, 'max_inner', cfg.maxInner));
cfg.tolInner = double(localOpt(runCfg, 'tol_inner', cfg.tolInner));
cfg.minInner = double(localOpt(runCfg, 'min_inner', cfg.minInner));
cfg.name = 'yuksel_problem_surface';

[cfg, problem] = localProblem(cfg, bcType, runCfg);
localValidate(cfg, problem);

threadsBefore = maxNumCompThreads();
threadCleanup = onCleanup(@() maxNumCompThreads(threadsBefore));
maxNumCompThreads(cfg.threads);

totalTic = tic;
mdl = localModel(cfg, problem);
NE = mdl.nele;
rho = cfg.rho0 * ones(NE,1);
flt = prepFilter(cfg.nelx, cfg.nely, cfg.rminEl);

n = cfg.n;
Nmax = cfg.Nmax;
Jcalc = n + Nmax;
cumInner = 0;
failureIterations = [];
eventLog = {};
hist = struct('omega',[],'N',[],'beta',[],'nInner',[],'cumInner',[], ...
    'innerConv',[],'lpFlag',[],'lpBackendIterations',[],'dxOuter',[], ...
    'vol',[],'tEig',[],'tGrad',[],'tInner',[],'degen',[],'multJ',[]);

if cfg.verbose
    fprintf('[OlhoffReproduced2007] %s, %dx%d, LxH=%gx%g, rmin=%g, move=%g\n', ...
        problem.label, cfg.nelx, cfg.nely, cfg.a, cfg.b, cfg.rminEl, cfg.move);
    fprintf('[OlhoffReproduced2007] inner solver: %s\n', route.description);
    if mdl.tipMassValue > 0
        fprintf('[OlhoffReproduced2007] point mass = %g on each right-mid translational DOF\n', ...
            mdl.tipMassValue);
    end
    fprintf('%4s %10s %10s %10s %3s %10s %7s %7s %8s %8s %8s\n', ...
        'it','omega1','omega2','omega3','N','sqrt(beta)', ...
        'inner','conv','maxdrho','volume',route.flagHeader);
end

for outer = 1:cfg.maxOuter
    eigTic = tic;
    [K,M] = localAssemble(mdl, rho, cfg.p, cfg.massInterp);
    [w,Phi,lam] = eigSolve(K, M, Jcalc, cfg.solver);
    tEig = toc(eigTic);

    N = 1;
    while n+N <= Jcalc-1 && abs(w(n+N)-w(n))/w(n) < cfg.tolMult
        N = N + 1;
    end
    J = n + N;
    multJ = (J+1 <= Jcalc) && abs(w(J+1)-w(J))/w(J) < cfg.tolMult;
    if N >= Nmax
        eventLog{end+1} = sprintf( ...
            'iter %d: detected N=%d >= Nmax=%d; multiplicity may be truncated', ...
            outer, N, Nmax); %#ok<AGROW>
    end
    if multJ
        eventLog{end+1} = sprintf( ...
            'iter %d: omega_J (J=%d) is multiple; equation (25b) is undefined', ...
            outer, J); %#ok<AGROW>
    end

    gradTic = tic;
    idx = n:(n+N-1);
    lamTild = mean(lam(idx));
    F = genGrad(mdl, rho, cfg.p, cfg.massInterp, Phi, lamTild, idx);
    FJ = genGrad(mdl, rho, cfg.p, cfg.massInterp, Phi, lam(J), J);
    fJJ = FJ(:,1,1);
    switch lower(cfg.filterMode)
        case 'diag'
            for j = 1:N
                F(:,j,j) = applyFilter(flt, rho, F(:,j,j));
            end
        case 'all'
            for s = 1:N
                for k = s:N
                    v = applyFilter(flt, rho, F(:,s,k));
                    F(:,s,k) = v;
                    F(:,k,s) = v;
                end
            end
        case 'none'
        otherwise
            error('topopt_olhoff_reproduced2007:FilterMode', ...
                'Unknown filter mode %s.', cfg.filterMode);
    end
    fJJ = applyFilter(flt, rho, fJJ);
    tGrad = toc(gradTic);

    innerTic = tic;
    ctx = struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J), ...
        'rho',rho,'rhomin',cfg.rhomin,'volfrac',cfg.volfrac, ...
        'move',cfg.move,'maxInner',cfg.maxInner,'tolInner',cfg.tolInner, ...
        'minInner',cfg.minInner,'offDiag',cfg.offDiag);
    if strcmpi(cfg.innerSolver,'lp')
        [drho,st] = innerLoopLP(ctx);
    else
        [drho,st] = innerLoop(ctx);
    end
    tInner = toc(innerTic);
    if isfield(st,'lpFlag'), lpFlag = st.lpFlag; else, lpFlag = NaN; end
    if isfield(st,'lpIterations')
        lpIterations = st.lpIterations;
    else
        lpIterations = NaN;
    end
    if strcmpi(cfg.innerSolver,'lp')
        if ~st.conv || lpFlag ~= 1 || any(~isfinite(drho))
            failureIterations(end+1) = outer; %#ok<AGROW>
            eventLog{end+1} = sprintf('iter %d: LP solve failed (flag=%d)', ...
                outer, lpFlag); %#ok<AGROW>
        end
    elseif any(~isfinite(drho))
        failureIterations(end+1) = outer; %#ok<AGROW>
        eventLog{end+1} = sprintf( ...
            'iter %d: MMA inner loop returned a nonfinite increment', ...
            outer); %#ok<AGROW>
    elseif ~st.conv
        % Reaching the sub-iterate cap is the documented behaviour of the
        % paper-literal route, not a solver breakdown: it is logged and left
        % out of INFO.failureIterations so that STATUS keeps its meaning.
        eventLog{end+1} = sprintf( ...
            'iter %d: MMA inner loop hit the %d sub-iterate cap without reaching tolInner=%g', ...
            outer, cfg.maxInner, cfg.tolInner); %#ok<AGROW>
    end

    % Preserve the imported OLHOFFOPT update semantics: the returned increment
    % is clipped to the admissible box even when the inner status is retained
    % as a failure in INFO.
    rho = min(1, max(cfg.rhomin, rho + drho));
    dxOuter = max(abs(drho));
    cumInner = cumInner + st.nInner;

    hist.omega(:,outer) = w(1:min(Jcalc,numel(w)));
    hist.N(outer) = N;
    hist.beta(outer) = st.beta;
    hist.nInner(outer) = st.nInner;
    hist.cumInner(outer) = cumInner;
    hist.innerConv(outer) = st.conv;
    hist.lpFlag(outer) = lpFlag;
    hist.lpBackendIterations(outer) = lpIterations;
    hist.dxOuter(outer) = dxOuter;
    hist.vol(outer) = mean(rho);
    hist.tEig(outer) = tEig;
    hist.tGrad(outer) = tGrad;
    hist.tInner(outer) = tInner;
    hist.degen(outer) = st.degenHits;
    hist.multJ(outer) = multJ;

    if cfg.verbose
        fprintf('%4d %10.3f %10.3f %10.3f %3d %10.3f %7d %7s %8.4f %8.4f %8s\n', ...
            outer, w(1), w(2), w(min(3,end)), N, sqrt(max(st.beta,0)), ...
            st.nInner, localYesNo(st.conv), dxOuter, mean(rho), ...
            localFlagText(lpFlag));
    end
    if dxOuter < cfg.tolOuter
        eventLog{end+1} = sprintf( ...
            'converged at iteration %d (max|drho|=%.3e)', outer, dxOuter); %#ok<AGROW>
        break
    end
end

[K,M] = localAssemble(mdl, rho, cfg.p, cfg.massInterp);
[omega,Phi,lambda] = eigSolve(K, M, Jcalc, cfg.solver);
modeTable = classifyModes(mdl, M, Phi, omega);
nOuter = numel(hist.N);
wallclock = toc(totalTic);

if ~isempty(failureIterations)
    status = 'SOLVER_FAILURE';
elseif nOuter > 0 && hist.dxOuter(end) < cfg.tolOuter
    status = 'CONVERGED';
else
    status = 'CAP_HIT';
end

info = struct();
info.cfg = cfg;
info.problem = problem;
info.route = route;
info.optimizer = route.id;
info.history = hist;
info.modeTable = modeTable;
info.lambda = lambda;
info.nOuter = nOuter;
info.wallclock = wallclock;
info.averageIterationTime = wallclock/max(nOuter,1);
info.status = status;
info.failureIterations = failureIterations;
info.log = eventLog;
info.model = mdl;
info.source = struct( ...
    'method',sprintf( ...
        'Du--Olhoff 2007 clean-room reproduction, %s route', route.label), ...
    'reproductionRoot',identity.root, ...
    'baselineConfiguration',sourceMeta.name, ...
    'extension','local cantilever boundary and constant nondesign point mass only');
end

function [cfg, route] = localOptimizer(cfg, runCfg)
% Select the Step 3 inner solver of the frozen reproduction.  The choice is
% carried in cfg.innerSolver exactly as OLHOFFOPT carries it, so the two
% entry points stay interchangeable.
requested = localOpt(runCfg, 'optimizer', 'lp');
key = lower(regexprep(char(string(requested)), '[-_ ]', ''));
offDiagGiven = isfield(runCfg,'off_diag') && ~isempty(runCfg.off_diag);
switch key
    case {'lp','linprog','krogolhoff','eq22'}
        cfg.innerSolver = 'lp';
        % INNERLOOPLP always imposes f_sk'drho = 0 (Eq. 22); there is no
        % coupled variant of it, so an explicit request for one is refused
        % rather than silently ignored.
        if offDiagGiven && logical(runCfg.off_diag)
            error('topopt_olhoff_reproduced2007:LpOffDiag', ...
                ['off_diag=true is not available on the LP route: Eq. (22) is ' ...
                 'built into innerLoopLP.  Use optimizer="mma" for the full ' ...
                 'Eq. (25d) coupling.']);
        end
        cfg.offDiag = false;
        route = struct('id','lp','label','Eq. (22) LP','flagHeader','LP', ...
            'description',['Eq. (22) LP route (Krog & Olhoff 1999), one ' ...
                           'linprog solve per outer iteration']);
    case {'mma','paper','papermma','eq25','eq25d'}
        cfg.innerSolver = 'mma';
        cfg.offDiag = logical(localOpt(runCfg, 'off_diag', true));
        if cfg.offDiag
            route = struct('id','mma','label','Eq. (25d) MMA', ...
                'flagHeader','-', ...
                'description',['paper-literal MMA inner loop on problem (25) ' ...
                               'with the full Eq. (25d) coupling']);
        else
            route = struct('id','mma','label','Eq. (22) MMA', ...
                'flagHeader','-', ...
                'description',['MMA inner loop with the Eq. (22) off-diagonal ' ...
                               'conditions imposed as inequality pairs']);
        end
    otherwise
        error('topopt_olhoff_reproduced2007:Optimizer', ...
            'Unsupported optimizer "%s". Use lp or mma.', char(string(requested)));
end
end

function value = localFlagText(flag)
if isnan(flag)
    value = '-';
else
    value = sprintf('%d', flag);
end
end

function [cfg, problem] = localProblem(cfg, bcType, runCfg)
key = lower(regexprep(char(string(bcType)), '[-_ ]', ''));
switch key
    case {'simply','simplysupported','ss'}
        cfg.a = 8;
        cfg.b = 1;
        cfg.bc = 'a';
        cfg.support = 'mid';
        cfg.axial = 'both';
        label = 'simply supported beam';
        tipMassFraction = 0;
    case {'fixedpinned','clampedsimply','cs'}
        cfg.a = 8;
        cfg.b = 1;
        cfg.bc = 'b';
        cfg.support = 'mid';
        cfg.axial = 'both';
        label = 'fixed--pinned beam';
        tipMassFraction = 0;
    case {'cantilever','cf'}
        cfg.a = 15;
        cfg.b = 10;
        cfg.bc = 'cantilever';
        cfg.support = 'face';
        cfg.axial = 'one';
        label = 'cantilever with right-mid-edge concentrated mass';
        tipMassFraction = double(localOpt(runCfg, 'tip_mass_fraction', 0.20));
    otherwise
        error('topopt_olhoff_reproduced2007:BoundaryCondition', ...
            'Unsupported bcType "%s". Use simply, fixedPinned, or cantilever.', ...
            char(string(bcType)));
end
problem = struct('id',key,'label',label,'length',cfg.a,'height',cfg.b, ...
    'tipMassFraction',tipMassFraction);
end

function localValidate(cfg, problem)
if cfg.rho0 < cfg.rhomin || cfg.rho0 > 1
    error('topopt_olhoff_reproduced2007:InitialDensity', ...
        'volfrac/rho0 must lie in [rho_min,1].');
end
if cfg.Nmax < 2 || cfg.n + cfg.Nmax >= 2*(cfg.nelx+1)*(cfg.nely+1)
    error('topopt_olhoff_reproduced2007:ModeBudget', ...
        'The requested eigenmode budget is not valid for this mesh.');
end
if ~strcmp(problem.id,'cantilever') && mod(cfg.nely,2) ~= 0
    error('topopt_olhoff_reproduced2007:MidSupport', ...
        'The mid-height support requires an even nely (got %d).', cfg.nely);
end
if problem.tipMassFraction < 0
    error('topopt_olhoff_reproduced2007:TipMass', ...
        'tip_mass_fraction must be nonnegative.');
end
if cfg.maxInner < 1 || cfg.minInner < 1 || cfg.minInner > cfg.maxInner
    error('topopt_olhoff_reproduced2007:InnerBudget', ...
        'Require 1 <= min_inner <= max_inner (got %g and %g).', ...
        cfg.minInner, cfg.maxInner);
end
if cfg.tolInner <= 0
    error('topopt_olhoff_reproduced2007:InnerTolerance', ...
        'tol_inner must be positive.');
end
end

function mdl = localModel(cfg, problem)
% Same Q4 forward model as reproduction2007/fem/model2D.m, extended only
% with a clamped-free boundary and a constant point mass.
nelx = cfg.nelx;
nely = cfg.nely;
dx = cfg.a/nelx;
dy = cfg.b/nely;
nele = nelx*nely;
nnode = (nelx+1)*(nely+1);
ndof = 2*nnode;

nodenrs = reshape(1:nnode, 1+nely, 1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1, nele, 1);
edofMat = repmat(edofVec,1,8) + ...
    repmat([0 1 2*nely+[2 3 0 1] -2 -1], nele, 1);
iK = reshape(kron(edofMat,ones(8,1))',64*nele,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nele,1);
[ely,elx] = ndgrid(1:nely,1:nelx);
cx = (elx(:)-0.5)*dx;
cy = (ely(:)-0.5)*dy;
[K0,M0] = elemMats2D(dx,dy,cfg.E,cfg.nu,cfg.rhom,cfg.t, ...
    cfg.elemType,cfg.massType);

leftCol = 1;
rightCol = nelx+1;
rowMid = round(nely/2)+1;
dofsOf = @(nodes,comp) 2*nodes(:)-2+comp;
clampFace = @(col) reshape([dofsOf(nodenrs(:,col),1); ...
    dofsOf(nodenrs(:,col),2)],[],1);

switch problem.id
    case {'simply','simplysupported','ss'}
        leftMid = nodenrs(rowMid,leftCol);
        rightMid = nodenrs(rowMid,rightCol);
        fixed = [dofsOf(leftMid,1); dofsOf(leftMid,2); ...
            dofsOf(rightMid,1); dofsOf(rightMid,2)];
    case {'fixedpinned','clampedsimply','cs'}
        rightMid = nodenrs(rowMid,rightCol);
        fixed = [clampFace(leftCol); dofsOf(rightMid,1); dofsOf(rightMid,2)];
    case 'cantilever'
        fixed = clampFace(leftCol);
    otherwise
        error('topopt_olhoff_reproduced2007:InternalBC', ...
            'Unhandled normalized boundary key %s.', problem.id);
end
fixed = unique(fixed(:));
free = setdiff((1:ndof)',fixed);

tipMassDofs = zeros(0,1);
tipMassValue = 0;
if problem.tipMassFraction > 0
    tipNode = nodenrs(rowMid,rightCol);
    tipMassDofs = [dofsOf(tipNode,1); dofsOf(tipNode,2)];
    permittedMass = cfg.volfrac*cfg.a*cfg.b*cfg.t*cfg.rhom;
    tipMassValue = problem.tipMassFraction*permittedMass;
end

mdl = struct('cfg',cfg,'nelx',nelx,'nely',nely,'dx',dx,'dy',dy, ...
    'nele',nele,'nnode',nnode,'ndof',ndof,'nodenrs',nodenrs, ...
    'edofMat',edofMat,'iK',iK,'jK',jK,'K0',K0,'M0',M0, ...
    'free',free,'fixed',fixed,'cx',cx,'cy',cy,'Ve',dx*dy*cfg.t, ...
    'tipMassDofs',tipMassDofs,'tipMassValue',tipMassValue);
end

function [K,M] = localAssemble(mdl, rho, penal, massInterp)
[K,M] = assemble2D(mdl, rho, penal, massInterp);
if mdl.tipMassValue <= 0
    return;
end
[isFree,reducedDofs] = ismember(mdl.tipMassDofs,mdl.free);
if ~all(isFree)
    error('topopt_olhoff_reproduced2007:ConstrainedTipMass', ...
        'A concentrated-mass DOF is constrained.');
end
nFree = numel(mdl.free);
M = M + sparse(reducedDofs,reducedDofs, ...
    mdl.tipMassValue*ones(numel(reducedDofs),1),nFree,nFree);
M = (M+M')/2;
end

function value = localOpt(options, name, defaultValue)
if isfield(options,name) && ~isempty(options.(name))
    value = options.(name);
else
    value = defaultValue;
end
end

function value = localYesNo(condition)
if condition
    value = 'yes';
else
    value = 'NO';
end
end
