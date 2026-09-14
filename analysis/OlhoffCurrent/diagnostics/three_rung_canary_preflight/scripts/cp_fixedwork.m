function fw = cp_fixedwork(nelx, nely, K)
%CP_FIXEDWORK  Part F: FIXED-WORK kernel timing at a canary's SAVED state.
%
%   fw = CP_FIXEDWORK(480, 60)   uses the state cp_run saved for that canary
%
%   Total solve time mixes cost-per-operation with number-of-operations-required.
%   This benchmark isolates the first.  It re-evaluates the three kernels K
%   times AT A FIXED DESIGN and NEVER updates it:
%
%       assembly + eigensolve      assemble2D + eigSolve
%       generalized gradients      genGrad x (2N+1) + the sensitivity filter
%       one MMA sub-problem        innerLoop, from an identical ctx each time
%
%   THE DESIGN IS NEVER ADVANCED.  drho is computed and discarded, rho is read
%   only, and an assertion refuses to return if rho changed.  Nothing written
%   here can reach a scientific run: this function produces no design, no
%   trajectory and no convergence claim.
%
%   K defaults to 5, with 2 discarded warm-up evaluations before each kernel.
%   Large enough for a stable median, far too small to constitute a campaign --
%   which is the intent.
%
%   See also CP_RUN.

if nargin < 3 || isempty(K), K = 5; end
WARM = 2;

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);

tag = sprintf('C%dx%d_three_rung', nelx, nely);
stateFile = fullfile(root,'evidence','three_rung_canary_preflight', ...
                     sprintf('%s_state.mat', tag));
assert(isfile(stateFile), 'cp_fixedwork:NoState', ...
    ['no saved state for %s.  This benchmark reads a state a canary already ' ...
     'produced; it never creates one.'], tag);
S = load(stateFile); st = S.state;
cfg = st.cfg; rho = st.rho;
g = @(p) olh.config.getPath(cfg,p);

probe = cp_hostprobe(sprintf('%s_fixedwork', tag), study);

% ---- rebuild exactly the solver's own computational state ---------------
flat = olh.config.toLegacy(cfg);
mdl  = model2D(flat);
NE   = mdl.nele;
assert(NE == st.NE, 'cp_fixedwork:MeshMismatch','saved state is not this mesh');
useMMA(g('optimizer.inner.variant'));

dyEl   = g('domain.b')/g('domain.mesh.nely');
rminEl = g('filter.radiusPhysical')/dyEl;
flt    = prepFilter(g('domain.mesh.nelx'), g('domain.mesh.nely'), rminEl);

n = g('eigen.targetMode'); Nmax = g('eigen.maxCluster'); Jcalc = n + Nmax;
pNow = g('material.stiffness.p');
massCfg = g('material.mass');
solver = g('eigen.solver');

% ================= kernel 1: assembly + eigensolve ======================
for k = 1:WARM
    [K1,M1] = assemble2D(mdl, rho, pNow, massCfg);
    [w,Phi,lam] = eigSolve(K1, M1, Jcalc, solver);
end
t = zeros(K,1);
for k = 1:K
    tc = tic;
    [K1,M1] = assemble2D(mdl, rho, pNow, massCfg);
    [w,Phi,lam] = eigSolve(K1, M1, Jcalc, solver);
    t(k) = toc(tc);
end
fw.eig = local_stat(t);

[N, ~] = olh.multi.detect(cfg, w, n, Jcalc, []);
J = n + N; idx = n:(n+N-1); lamTild = lam(n);
gctx = struct('mdl',mdl,'rho',rho,'p',pNow,'mass',massCfg,'Phi',Phi, ...
              'lam',lam,'idx',idx,'lamTild',lamTild,'J',J,'N',N,'flt',flt);

% ================= kernel 2: generalized gradients ======================
for k = 1:WARM, local_grads(gctx); end
t = zeros(K,1);
for k = 1:K, tc = tic; [F, fJJ] = local_grads(gctx); t(k) = toc(tc); end
fw.grad = local_stat(t);

% ---- the ctx the inner solver would have been handed -------------------
ctx = struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J), ...
             'rho',rho,'rhomin',g('design.minimum'), ...
             'volfrac',g('design.volumeFraction'), ...
             'move',cfg.move.levels(end), ...
             'maxInner',g('optimizer.inner.maxIterations'), ...
             'tolInner',g('optimizer.inner.tolerance'), ...
             'minInner',g('optimizer.inner.minIterations'), ...
             'offDiag',g('multiplicity.offDiagonal'), ...
             'dOff',lam(idx) - lam(idx(1)));

% ================= kernel 3: ONE MMA sub-problem ========================
% drho is computed and DISCARDED.  rho is not touched.
for k = 1:WARM, [~, ~] = innerLoop(ctx); end
t = zeros(K,1); nin = zeros(K,1);
for k = 1:K
    tc = tic; [drho, sti] = innerLoop(ctx); t(k) = toc(tc);
    nin(k) = sti.nInner;
end
fw.inner = local_stat(t);
fw.inner.nInner = median(nin);
fw.inner.s_per_mma_step = fw.inner.median/fw.inner.nInner;
clear drho

assert(isequal(rho, st.rho), 'cp_fixedwork:DesignMutated', ...
    'the fixed-work benchmark must not change the design');

fw.mesh = [nelx nely]; fw.NE = NE; fw.K = K; fw.warmup = WARM;
fw.freeDOF = mdl.ndof - numel(mdl.fixed);
fw.N = N; fw.J = J; fw.Jcalc = Jcalc;
fw.move_used = ctx.move;
fw.omega_at_state = w(1:min(5,numel(w)));
fw.cfgHash = olhoffcurrent_config_hash(cfg);
fw.nOuter_of_source_run = st.nOuter;
fw.host = probe;
fw.matlab = version;
fw.when = char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ss'));

outFile = fullfile(study,'evidence',sprintf('fixedwork_%dx%d.json', nelx, nely));
fid = fopen(outFile,'w'); fprintf(fid,'%s', jsonencode(fw,'PrettyPrint',true)); fclose(fid);
fprintf(['[cp_fixedwork] %dx%d  eig %.4fs  grad %.4fs  mma %.4fs ' ...
         '(%d steps, %.5f s/step)\n'], nelx, nely, fw.eig.median, fw.grad.median, ...
         fw.inner.median, fw.inner.nInner, fw.inner.s_per_mma_step);
end

% =========================================================================
function [F, fJJ] = local_grads(c)
%LOCAL_GRADS  One complete sensitivity evaluation, exactly as olhoffSolve does
%   it on the frozen path: the cluster gradients, the diagonal-offset rebuild,
%   the next-mode gradient, then the Sigmund sensitivity filter over every f_sk.
F = genGrad(c.mdl, c.rho, c.p, c.mass, c.Phi, c.lamTild, c.idx);
for j = 1:c.N
    Gj = genGrad(c.mdl, c.rho, c.p, c.mass, c.Phi, c.lam(c.idx(j)), c.idx(j));
    F(:,j,j) = Gj(:,1,1);
end
FJ  = genGrad(c.mdl, c.rho, c.p, c.mass, c.Phi, c.lam(c.J), c.J);
fJJ = FJ(:,1,1);
for s = 1:c.N
    for k = s:c.N
        v = applyFilter(c.flt, c.rho, F(:,s,k));
        F(:,s,k) = v;  F(:,k,s) = v;
    end
end
fJJ = applyFilter(c.flt, c.rho, fJJ);
end

function s = local_stat(t)
s = struct('median',median(t),'min',min(t),'max',max(t), ...
           'mean',mean(t),'std',std(t),'n',numel(t),'samples',t(:).');
s.spread_pct = 100*(s.max - s.min)/max(s.median, eps);
end
