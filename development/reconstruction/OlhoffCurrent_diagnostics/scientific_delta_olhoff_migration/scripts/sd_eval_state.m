function R = sd_eval_state(cfg, rho, boxes)
%SD_EVAL_STATE  Offline replay of olhoffSolve steps 1-3 at a FROZEN state.
%
%   R = SD_EVAL_STATE(cfg, rho, boxes) uses whatever implementation is on the
%   path (target +impl or the source snapshot; the caller installs it) and the
%   given effective configuration.  The statements below are copied from the
%   respective olhoffSolve.m (source 6b08708 / target +impl) for the sensitivity-
%   filtered, unprojected, fixed-p, increment-variable path that both C480 and
%   S480 execute; the only branch is how each implementation is told the
%   stiffness law (numeric p for the target, a struct for the source).
%
%   boxes : cell array of move boxes (scalar or NE x 1) for which ONE full inner
%           problem-(25) solve is performed.  Nothing is written back to rho.
g = @(p) olh.config.getPath(cfg, p);
isSource = ~isempty(which('olh.material.stiffnessInterpolation'));
useMMA(g('optimizer.inner.variant'));
flat = olh.config.toLegacy(cfg);
mdl  = model2D(flat);
NE   = mdl.nele;
rho  = rho(:);
assert(numel(rho) == NE, 'sd:eval:size', 'state size %d vs NE %d', numel(rho), NE);
rminPhys = g('filter.radiusPhysical');
if ~isempty(rminPhys) && rminPhys > 0
    rminEl = rminPhys/(g('domain.b')/g('domain.mesh.nely'));
else
    rminEl = g('filter.radiusElements');
end
flt   = prepFilter(g('domain.mesh.nelx'), g('domain.mesh.nely'), rminEl);
n     = g('eigen.targetMode');
Jcalc = n + g('eigen.maxCluster');
pNow  = g('material.stiffness.p');
massNowCfg = g('material.mass');
if isSource
    stiffNow = struct('model', g('material.stiffness.model'), 'p', pNow, ...
                      'linearBelow', g('material.stiffness.linearBelow'));
    eigOpts  = struct('tol',g('eigen.tolerance'),'maxit',g('eigen.maxIterations'), ...
                      'pFactor',g('eigen.krylovFactor'));
else
    stiffNow = pNow;
end

% ---- step 1 --------------------------------------------------------------
t = tic;
[K,M] = assemble2D(mdl, rho, stiffNow, massNowCfg);
if isSource
    [w, Phi, lam] = eigSolve(K, M, Jcalc, g('eigen.solver'), [], eigOpts);
else
    [w, Phi, lam] = eigSolve(K, M, Jcalc, g('eigen.solver'));
end
R.tEig = toc(t);
[N, ~] = olh.multi.detect(cfg, w, n, Jcalc, []);
J = n + N;
multJ = (J+1 <= Jcalc) && abs(w(J+1)-w(J))/w(J) < g('multiplicity.tolerance');

% ---- step 2 --------------------------------------------------------------
idx     = n:(n+N-1);
lamTild = lam(n);
F       = genGrad(mdl, rho, stiffNow, massNowCfg, Phi, lamTild, idx);
useOff  = g('multiplicity.diagonalOffsets');
if useOff
    for j = 1:N
        Gj = genGrad(mdl, rho, stiffNow, massNowCfg, Phi, lam(idx(j)), idx(j));
        F(:,j,j) = Gj(:,1,1);
    end
    dOff = lam(idx) - lam(idx(1));
else
    dOff = [];
end
FJ  = genGrad(mdl, rho, stiffNow, massNowCfg, Phi, lam(J), J);
fJJ = FJ(:,1,1);
Fraw = F; fJJraw = fJJ;
assert(strcmp(g('filter.type'),'sensitivity') && strcmp(g('filter.applyTo'),'all') && ~g('projection.enabled'), ...
    'sd:eval:path', 'evaluator covers only the unprojected all-f_sk sensitivity-filter path');
for s = 1:N
    for k = s:N
        v = applyFilter(flt, rho, F(:,s,k));
        F(:,s,k) = v;  F(:,k,s) = v;
    end
end
fJJ = applyFilter(flt, rho, fJJ);

% ---- element energies of the lowest modes (Part 18), post hoc -------------
nm = min(5, size(Phi,2));
Ufull = zeros(mdl.ndof, nm); Ufull(mdl.free,:) = Phi(:,1:nm);
if isSource
    gK = olh.material.stiffnessInterpolation(rho, stiffNow);
else
    gK = rho.^pNow;
end
gM = massScale(rho, massNowCfg);
Ekin = zeros(NE, nm); Estr = zeros(NE, nm);
for j = 1:nm
    Ue = reshape(Ufull(mdl.edofMat(:), j), NE, 8);
    Ekin(:,j) = gM(:) .* sum((Ue*mdl.M0).*Ue, 2);
    Estr(:,j) = gK(:) .* sum((Ue*mdl.K0).*Ue, 2);
end

% ---- problem (25) rows at drho = 0 and full inner solves ------------------
offDiag = g('multiplicity.offDiagonal');
base = struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J), ...
    'rho',rho,'rhomin',g('design.minimum'),'volfrac',g('design.volumeFraction'), ...
    'maxInner',g('optimizer.inner.maxIterations'), ...
    'tolInner',g('optimizer.inner.tolerance'), ...
    'minInner',g('optimizer.inner.minIterations'), ...
    'offDiag',offDiag,'dOff',dOff);
if isSource, base.asyOuter = strcmp(g('optimizer.inner.asymptoteHistory'),'outer'); end
[R.rows_fval, R.rows_dfdx] = sd_rows(base);
R.inner = struct('box',{},'drho',{},'nInner',{},'conv',{},'beta',{},'relHist',{},'dxHist',{},'t',{}, ...
                 'dlamPred',{});
for b = 1:numel(boxes)
    ctx = base; ctx.move = boxes{b};
    t = tic;
    [drho, st] = innerLoop(ctx);
    tt = toc(t);
    R.inner(b) = struct('box', boxes{b}, 'drho', drho, 'nInner', st.nInner, 'conv', st.conv, ...
        'beta', st.beta, 'relHist', st.relHist, 'dxHist', st.dxHist, 't', tt, ...
        'dlamPred', deltaLambda(F, drho, dOff));
end

[iK,jK,vK] = find(K); [iM,jM,vM] = find(M);
R.isSource = isSource; R.impl = which('olhoffSolve'); R.cfgHash = sd_cfghash_any(cfg);
R.stiffness = g('material.stiffness.model'); R.mass = massNowCfg.model;
R.rho_sha256 = sd_sha256_double(rho);
R.K_sha256 = sd_sha256_double([iK; jK; vK]); R.M_sha256 = sd_sha256_double([iM; jM; vM]);
R.K_fro = norm(vK); R.M_fro = norm(vM); R.K_nnz = nnz(K); R.M_nnz = nnz(M);
R.omega = w(:); R.lam = lam(:); R.N = N; R.J = J; R.multJ = multJ; R.dOff = dOff(:);
R.lamTild = lamTild; R.rminEl = rminEl;
R.Fraw = Fraw; R.fJJraw = fJJraw; R.Ffilt = F; R.fJJfilt = fJJ;
R.Phi = Phi(:,1:nm); R.Ekin = Ekin; R.Estr = Estr; R.gK = gK; R.gM = gM;
end

function h = sd_cfghash_any(cfg)
L = sd_flatten(cfg);
keep = ~startsWith(L(:,1), 'provenance.') & ~strcmp(L(:,1), 'runtime.name');
s = strjoin(cellfun(@(p,v) [p '=' mat2str_any(v)], L(keep,1), L(keep,2), 'UniformOutput', false), newline);
md = java.security.MessageDigest.getInstance('SHA-256'); md.update(uint8(s));
d = typecast(md.digest(), 'uint8'); h = lower(reshape(dec2hex(d, 2).', 1, []));
end

function s = mat2str_any(v)
if ischar(v) || isstring(v), s = char(v);
elseif isnumeric(v) || islogical(v), s = mat2str(v, 17);
elseif iscell(v), s = strjoin(cellfun(@mat2str_any, v, 'UniformOutput', false), '|');
else, s = class(v);
end
end
