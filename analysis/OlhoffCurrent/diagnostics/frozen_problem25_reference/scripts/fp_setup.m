function S = fp_setup()
%FP_SETUP  Load the authoritative frozen 480 state and build read-only objects.
%
%   READ-ONLY.  Nothing here can write a density.  The production path guard
%   is installed and carried in S.guard so it stays alive for every caller.
%   The trajectory is read directly from the authoritative evidence file, not
%   from any derived cache of an earlier audit.

persistent C
if ~isempty(C), S = C; return, end

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));          % analysis/OlhoffCurrent
addpath(root); addpath(here);
guard = olhoffcurrent_paths();
maxNumCompThreads(1);

traj = fullfile(root,'evidence','three_rung_canary_preflight', ...
                'C480x60_three_rung_trajectory.mat');
assert(isfile(traj), 'fp_setup:NoTrajectory', 'frozen 480 trajectory absent');
T = load(traj, 'RHO','DRHO','hist','cfg','meta','exh');
n = numel(T.hist.N);
assert(n == 386, 'fp_setup:WrongOuter', 'expected 386 outer, got %d', n);

S = struct();
S.study   = study;
S.root    = root;
S.rho386  = T.RHO(:,end);         % authoritative frozen endpoint
S.rho385  = T.RHO(:,end-1);       % density the LAST subproblem was built at
S.drho386 = T.DRHO(:,end);        % production P19 increment
S.cfg     = T.cfg;
S.nOuter  = n;
S.hist    = struct('nInner',T.hist.nInner(:),'move',T.hist.move(:), ...
                   'stage',T.hist.stage(:),'beta',T.hist.beta(:), ...
                   'gap12',T.hist.gap12(:),'multJ',T.hist.multJ(:), ...
                   'N',T.hist.N(:),'omega',T.hist.omega, ...
                   'innerConv',T.hist.innerConv(:),'vol',T.hist.vol(:));
S.exh      = T.exh;
S.implTree = T.meta.implTree;
S.cfgHash  = T.meta.cfgHash;
clear T

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
