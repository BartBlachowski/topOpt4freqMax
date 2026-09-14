function nFail = test_modules()
%TEST_MODULES  The canonical dispatch modules against the legacy ones, and the
%   filter / projection / field-propagation behaviour.
nFail = 0;
rng(7);

% =====================================================================
% 1. multiplicity: olh.multi.detect == algo/multRule for every method
% =====================================================================
methods = {'binary','latch','hysteresis','subspace'};
for m = 1:numel(methods)
    stA = []; stB = [];  agree = true;
    cfg = olh.config.resolve('duOlhoffFrozenM4','multiplicity.method',methods{m});
    flat = olh.config.toLegacy(cfg);
    for it = 1:200
        % spectra spanning separated, near-degenerate and exactly degenerate
        w = sort(abs([100; 100*(1+10^(-1-4*rand)); 300*rand+150; 400; 500]));
        [NA, stA] = olh.multi.detect(cfg, w, 1, 5, stA);
        [NB, stB] = multRule(flat, w, 1, 5, stB);
        if NA ~= NB, agree = false; break; end
    end
    nFail = nFail + local_check(sprintf('multiplicity %-11s canonical == legacy over 200 spectra', ...
        methods{m}), agree);
end

% =====================================================================
% 2. move policy: olh.move.limit == algo/moveControl for every policy
% =====================================================================
policies = {'fixed','geometric','ladder','trustRatio'};
for m = 1:numel(policies)
    cfg = olh.config.resolve('duOlhoffFrozenM4','move.policy',policies{m});
    flat = olh.config.toLegacy(cfg);
    stA = []; stB = [];  agree = true;
    hist = struct('N',[],'beta',[],'dxNorm2',[],'omega',[],'move',[]);
    for it = 1:120
        hist.N(it)       = 1 + (it > 20);
        hist.beta(it)    = 1e4*(1 - exp(-it/25)) + 30*sin(it);   % rising, oscillating
        hist.dxNorm2(it) = 2*exp(-it/40) + 0.05*cos(it);
        hist.omega(1,it) = 90 + it/10;
        [mvA, stA] = olh.move.limit(cfg, it, hist, stA);
        [mvB, stB] = moveControl(flat, it, hist, stB);
        hist.move(it) = mvA;
        if ~isequal(typecast(mvA,'uint8'),typecast(mvB,'uint8')), agree = false; break; end
        if isfield(stA,'stage') && isfield(stB,'stage') && stA.stage ~= stB.stage
            agree = false; break
        end
    end
    nFail = nFail + local_check(sprintf('move %-11s canonical == legacy bitwise over 120 iterations', ...
        policies{m}), agree);
end
% and the alternative stall signal
cfg  = olh.config.resolve('duOlhoffFrozenM4','move.continuation.signal','designRms');
flat = olh.config.toLegacy(cfg);
nFail = nFail + local_check('move signal designRms maps to legacy s2Signal=''drms''', ...
    strcmp(flat.s2Signal,'drms'));

% =====================================================================
% 3. filtering: the two formulations are genuinely different operators
% =====================================================================
flt = prepFilter(20, 10, 2.0);
x   = rand(200,1)*0.9 + 0.05;
df  = randn(200,1);
sens = applyFilter(flt, x, df);
[~, sChain] = projDensityField(flt.H, flt.Hs, x, 0, 0.5, 1e-3);
dens = projChain(flt.H, flt.Hs, sChain, df);
nFail = nFail + local_check('sensitivity and density filtering are different operators', ...
    norm(sens - dens) > 1e-6*norm(sens));
nFail = nFail + local_check('sensitivity filter preserves the sum of rho.*df (top88 form)', ...
    abs(sum(flt.Hs.*max(1e-3,x).*sens) - sum(flt.H*(x.*df))) < 1e-8*abs(sum(flt.H*(x.*df))));

% =====================================================================
% 4. projection: identity at beta=0, mapping, derivative, monotone schedule
% =====================================================================
xv = linspace(0,1,101).';
[P, dP] = projectDensity(xv, 0, 0.5);
nFail = nFail + local_check('projection at beta=0 is the EXACT identity', ...
    isequal(typecast(P,'uint8'),typecast(xv,'uint8')) && all(dP==1));
for b = [1 2 4 8 16]
    [P, dP] = projectDensity(xv, b, 0.5);
    h = 1e-6;
    fd = (projectDensity(xv+h,b,0.5) - projectDensity(xv-h,b,0.5))/(2*h);
    rel = max(abs(dP-fd)./max(abs(fd),1e-9));
    ok = rel < 1e-5 && abs(P(1)) < 1e-12 && abs(P(end)-1) < 1e-12 && all(diff(P) > -1e-12);
    nFail = nFail + local_check(sprintf('projection beta=%-2d: P(0)=0, P(1)=1, monotone, dP matches FD (%.1e)', ...
        b, rel), ok);
end
% the three-field map and its chain rule
z = rand(200,1);
[rp, sC] = projDensityField(flt.H, flt.Hs, z, 4, 0.5, 1e-3);
nFail = nFail + local_check('physical density stays inside [rhomin,1]', ...
    all(rp >= 1e-3 - 1e-12) && all(rp <= 1 + 1e-12));
% chain rule against a finite difference of a scalar functional
w = randn(200,1);
J  = @(zz) w.'*local_rhoOf(flt, zz, 4, 0.5, 1e-3);
gz = projChain(flt.H, flt.Hs, sC, w);
h = 1e-6;  e = zeros(200,1);  err = 0;
for i = [3 47 111 199]
    e(:) = 0; e(i) = h;
    err = max(err, abs(gz(i) - (J(z+e)-J(z-e))/(2*h))/max(abs(gz(i)),1e-9));
end
nFail = nFail + local_check(sprintf('projection chain rule matches FD (max rel err %.1e)', err), err < 1e-5);

% =====================================================================
% 5. field propagation: design vs filtered vs physical
% =====================================================================
cfgP = olh.config.resolve('projected','runtime.maxOuter',3,'runtime.diagnostics',true);
resP = olhoffSolve(cfgP);
nFail = nFail + local_check('projected run exposes z, zTilde and rhoPhys separately', ...
    isfield(resP,'z') && isfield(resP,'zTilde') && isfield(resP,'rhoPhys'));
nFail = nFail + local_check('res.rho IS the physical density the FE model used', ...
    isequal(typecast(resP.rho,'uint8'),typecast(resP.rhoPhys,'uint8')));
nFail = nFail + local_check('the three fields are genuinely distinct under projection', ...
    norm(resP.z-resP.zTilde) > 1e-9 && norm(resP.zTilde-resP.rhoPhys) > 1e-9);
nFail = nFail + local_check('hist.dxPhys2 is recorded under projection', ...
    all(~isnan(resP.hist.dxPhys2)));

cfgF = olh.config.resolve('duOlhoffFrozenM4','runtime.maxOuter',3,'runtime.diagnostics',true);
resF = olhoffSolve(cfgF);
nFail = nFail + local_check('without projection the design variable IS the density', ...
    ~isfield(resF,'z') && all(isnan(resF.hist.dxPhys2)));

% =====================================================================
% 6. status precedence
% =====================================================================
nFail = nFail + local_check('a run stopped by the cap reports CAP_HIT', ...
    strcmp(resF.status,'CAP_HIT'));
% A run that genuinely meets the criterion must report CONVERGED, not CAP_HIT.
% A sub-160x20 mesh is used deliberately: this is a SOFTWARE MECHANICS test of
% the status logic, not scientific evidence, and Phase 7 permits that.
ws = warning('off','olh:config:suspicious');
cfgC = olh.config.resolve('duOlhoffFrozenM4', ...
    'domain.mesh.nelx',40,'domain.mesh.nely',10, ...
    'stop.toleranceRule','explicit','stop.tolerance',5, ...
    'runtime.maxOuter',50);
warning(ws);
resC = olhoffSolve(cfgC);
nFail = nFail + local_check(sprintf('a run meeting the criterion reports CONVERGED (got %s at outer %d of 50)', ...
    resC.status, resC.nOuter), strcmp(resC.status,'CONVERGED') && resC.nOuter < 50);
nFail = nFail + local_check('CONVERGED runs log the convergence line exactly once', ...
    sum(contains(resC.log,'converged at outer iteration')) == 1);
end
function r = local_rhoOf(flt, z, b, eta, rmin)
r = projDensityField(flt.H, flt.Hs, z, b, eta, rmin);
end
function n = local_check(name, cond)
if cond
    fprintf('  ok   %s\n', name); n = 0;
else
    fprintf('  FAIL %s\n', name); n = 1;
end
end
