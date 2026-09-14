function S = fi_setup()
%FI_SETUP  Load the frozen 480 state and build every read-only object needed.
%
%   Loads nothing that can be written back.  Builds the FE model and the filter
%   exactly as olhoffSolve does, so every evaluation in this audit uses the
%   production code path.  No density is ever updated by anything downstream.

persistent C
if ~isempty(C), S = C; return, end

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
addpath(fullfile(root,'diagnostics','three_rung_promotion_validation_retry1','scripts'));
% The path guard is an onCleanup object: it MUST outlive this function or the
% production path is torn down the moment fi_setup returns.  It is carried in
% the returned struct (and in the persistent cache) so every caller keeps it
% alive.  It is never saved to disk.
guard = olhoffcurrent_paths();
maxNumCompThreads(1);

cacheFile = fullfile(study,'evaluations','frozen_480_state.mat');
if isfile(cacheFile)
    L = load(cacheFile);
    S = L.S;
else
    traj = fullfile(root,'evidence','three_rung_canary_preflight', ...
                    'C480x60_three_rung_trajectory.mat');
    assert(isfile(traj), 'fi_setup:NoTrajectory', 'frozen 480 trajectory absent');
    T = load(traj, 'RHO','DRHO','hist','cfg','meta','exh');
    n = numel(T.hist.N);
    assert(n == 386, 'fi_setup:WrongOuter', 'expected 386 outer, got %d', n);
    S = struct();
    S.rho386 = T.RHO(:,end);          % authoritative frozen endpoint
    S.rho385 = T.RHO(:,end-1);        % the density the LAST subproblem was built at
    S.drho386 = T.DRHO(:,end);        % the increment that run actually produced
    S.cfg = T.cfg;
    S.nOuter = n;
    S.hist = struct('nInner',T.hist.nInner(:),'move',T.hist.move(:), ...
                    'stage',T.hist.stage(:),'beta',T.hist.beta(:), ...
                    'gap12',T.hist.gap12(:),'multJ',T.hist.multJ(:), ...
                    'N',T.hist.N(:),'omega',T.hist.omega, ...
                    'innerConv',T.hist.innerConv(:),'vol',T.hist.vol(:), ...
                    'dxNorm2',T.hist.dxNorm2(:),'dxOuter',T.hist.dxOuter(:));
    S.exh = T.exh;
    S.implTree = T.meta.implTree;
    S.cfgHash  = T.meta.cfgHash;
    if ~isfolder(fileparts(cacheFile)), mkdir(fileparts(cacheFile)); end
    save(cacheFile,'S','-v7.3');
end

cfg = S.cfg;
g = @(p) olh.config.getPath(cfg,p);
S.g = g;
S.flat = olh.config.toLegacy(cfg);
S.mdl  = model2D(S.flat);
S.NE   = S.mdl.nele;
S.nelx = g('domain.mesh.nelx');  S.nely = g('domain.mesh.nely');
S.p    = g('material.stiffness.p');
S.massCfg = g('material.mass');
S.rhomin  = g('design.minimum');
S.volfrac = g('design.volumeFraction');
S.n    = g('eigen.targetMode');
S.Nmax = g('eigen.maxCluster');
S.Jcalc= S.n + S.Nmax;
S.solver = g('eigen.solver');
S.move = S.hist.move(end);
S.dyEl = g('domain.b')/S.nely;
S.rminEl = g('filter.radiusPhysical')/S.dyEl;
S.flt = prepFilter(S.nelx, S.nely, S.rminEl);
S.Vtot = S.volfrac*S.NE;
useMMA(g('optimizer.inner.variant'));

S.guard = guard;
C = S;
end
