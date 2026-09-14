function cs_endpoint_spectral(which)
%CS_ENDPOINT_SPECTRAL  Part 10 inputs: frozen-state spectral quantities at an
%   endpoint.  OFFLINE, NO OPTIMIZATION.  The expressions are the spectral part
%   of gray_kkt_forensic_audit/scripts/frozen_evaluate.m (no finite differences).
%
%   which = 'control'   : control rho386; must reproduce the retained spectral_480.mat
%   which = 'treatment' : treatment final rho
S = cs_setup();
guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);
nx = 480; ny = 60;
switch which
    case 'control'
        mf = matfile(S.controlTraj); cfg = mf.cfg; rho = mf.RHO(:,386);
    case 'control14'
        mf = matfile(S.controlTraj); cfg = mf.cfg; rho = mf.RHO(:,14);
    case 'treatment'
        st = load(fullfile(S.evDir,'C480x60_socp_state.mat'),'state'); cfg = st.state.cfg; rho = st.state.rho;
    otherwise
        error('which');
end
frozen = rho;
flat = olh.config.toLegacy(cfg); mdl = model2D(flat); p = cfg.material.stiffness.p; mass = cfg.material.mass;
flt = prepFilter(nx, ny, cfg.filter.radiusPhysical/(1/ny));
[K,M] = assemble2D(mdl, rho, p, mass); [omega,Phi,lam] = eigSolve(K, M, 5, cfg.eigen.solver);
[N,~] = olh.multi.detect(cfg, omega, 1, 5, []); assert(N == 2); idx = 1:N; J = N+1;
Fraw = genGrad(mdl, rho, p, mass, Phi, lam(1), idx);
GK = genGrad(mdl, rho, p, mass, Phi, 0, idx);
for j = 1:N
    Gj = genGrad(mdl, rho, p, mass, Phi, lam(j), j); Fraw(:,j,j) = Gj(:,1,1);
end
GM = Fraw - GK; Ffiltered = Fraw;
for s = 1:N, for t = s:N
    ff = applyFilter(flt, rho, Fraw(:,s,t)); Ffiltered(:,s,t) = ff; Ffiltered(:,t,s) = ff;
end, end
FJ = genGrad(mdl, rho, p, mass, Phi, lam(J), J); fJraw = FJ(:,1,1); fJfiltered = applyFilter(flt, rho, fJraw);
dOff = lam(idx) - lam(1); [~,draw,Vraw] = deltaLambda(Fraw, zeros(size(rho)), dOff);
[~,dfiltered,Vfiltered] = deltaLambda(Ffiltered, zeros(size(rho)), dOff);
massOrth = norm(Phi'*M*Phi - eye(5), 'fro'); eigResidual = zeros(5,1);
for j = 1:5, eigResidual(j) = norm(K*Phi(:,j) - lam(j)*M*Phi(:,j))/(norm(K*Phi(:,j)) + norm(lam(j)*M*Phi(:,j))); end
gK = GK(:,1,1); gM = GM(:,1,1); gRaw = draw(:,1); gFiltered = dfiltered(:,1);
assert(isequal(rho, frozen), 'frozen state changed');
out = fullfile(S.study, 'evaluations', sprintf('spectral_%s.mat', which));
save(out, 'rho','omega','lam','Fraw','Ffiltered','GK','GM','fJraw','fJfiltered','gK','gM','gRaw','gFiltered', ...
    'massOrth','eigResidual','Vraw','Vfiltered','dOff','-v7');
fprintf('SPECTRAL %s omega1 %.14g gap12 %.6g\n', which, omega(1), (omega(2)-omega(1))/omega(1));
if strcmp(which, 'control')
    R = load(fullfile(S.root,'diagnostics','gray_kkt_forensic_audit','evaluations','spectral_480.mat'));
    c = struct();
    c.rho = isequal(R.rho(:), rho(:));
    rel = @(a,b) max(abs(a(:)-b(:)))/max(max(abs(b(:))), realmin);
    c.lam_rel = rel(lam, R.lam); c.gRaw_rel = rel(gRaw, R.gRaw); c.gFiltered_rel = rel(gFiltered, R.gFiltered);
    c.omega_rel = rel(omega, R.omega);
    c.pass = c.rho && c.lam_rel <= 1e-12 && c.gRaw_rel <= 1e-12 && c.gFiltered_rel <= 1e-12;
    cs_json(fullfile(S.study,'evaluations','spectral_control_reproduction.json'), c);
    disp(c);
end
end
