function out = fp_state()
%FP_STATE  Part 1: authoritative frozen 480 state identity and exact context.
%
%   Recovers the frozen state, verifies every preregistered hash, re-evaluates
%   the FE problem at rho385 through the PRODUCTION functions to rebuild the
%   inner-loop context ctx, and compares it with the ctx retained by the prior
%   audit.  Saves evaluations/frozen_ctx.mat, the single frozen input of every
%   later script.  READ-ONLY with respect to every density.
S = fp_setup();
ev = fullfile(S.study,'evaluations');

expect = struct( ...
    'rho386','0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60', ...
    'rho385','9b1443e508a6e4ecb5288d30167c7258d837be127aa3a9a42d8cc092240f3ded', ...
    'drho386','0b12cd7f9ae32decc6e95bf63e89fa7ca4530d13ec6dc7d815b618d46af8a9fd', ...
    'cfgHash','03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e', ...
    'implTree','edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb', ...
    'nOuter',386,'stage',3,'move',0.01,'N',2,'multJ',0,'nInner',19);

o = struct();
o.rho386_sha256  = fp_hash(S.rho386);
o.rho385_sha256  = fp_hash(S.rho385);
o.drho386_sha256 = fp_hash(S.drho386);
o.cfgHash          = olhoffcurrent_config_hash(S.cfg);
o.cfgHash_recorded = S.cfgHash;
sm = olhoffcurrent_source_manifest('Verify',true);
o.implTree = sm.treeHash;  o.implTree_ok = sm.ok;
o.implTree_recorded = S.implTree;
o.nOuter = S.nOuter;
o.stage  = S.hist.stage(end);
o.move   = S.hist.move(end);
o.N_hist = S.hist.N(end);
o.multJ  = S.hist.multJ(end);
o.nInner_final = S.hist.nInner(end);
o.innerConv_final = S.hist.innerConv(end);
o.beta_recorded_386 = S.hist.beta(end);
o.omega_hist_at_rho385 = S.hist.omega(:,end).';
o.gap12_hist_at_rho385 = S.hist.gap12(end);
o.rhomin = S.rhomin; o.volfrac = S.volfrac; o.Vtot = S.Vtot;
o.rho385_minmax = [min(S.rho385) max(S.rho385)];
o.rho385_mean = mean(S.rho385);
o.controller = struct('signal',S.g('move.continuation.signal'), ...
    'stopRule',S.g('stop.rule'),'levels',S.g('move.levels'), ...
    'terminalBranch',S.exh.terminalBranch,'terminalDeclIter',S.exh.terminalDeclIter);
o.multiplicity = struct('method',S.g('multiplicity.method'), ...
    'subspaceSize',S.g('multiplicity.subspaceSize'), ...
    'tolerance',S.g('multiplicity.tolerance'), ...
    'diagonalOffsets',S.g('multiplicity.diagonalOffsets'), ...
    'offDiagonal',S.g('multiplicity.offDiagonal'));
o.filter = struct('type',S.g('filter.type'),'applyTo',S.g('filter.applyTo'), ...
    'radiusPhysical',S.g('filter.radiusPhysical'),'rminEl',S.rminEl);
o.mma = struct('variant',S.g('optimizer.inner.variant'), ...
    'tolInner',S.g('optimizer.inner.tolerance'), ...
    'minInner',S.g('optimizer.inner.minIterations'), ...
    'maxInner',S.g('optimizer.inner.maxIterations'), ...
    'a0',1,'a',0,'c',1000,'d',0);

% ---- fresh FE evaluation at rho385 through the production functions -----
rho = S.rho385;
[K,M] = assemble2D(S.mdl, rho, S.p, S.massCfg);
[w, Phi, lam] = eigSolve(K, M, S.Jcalc, S.solver);
[N, ~] = olh.multi.detect(S.cfg, w, S.n, S.Jcalc, []);
idx = S.n:(S.n+N-1);  J = S.n + N;
lamTild = lam(S.n);
F = genGrad(S.mdl, rho, S.p, S.massCfg, Phi, lamTild, idx);
useOff = S.g('multiplicity.diagonalOffsets');
if useOff
    for j = 1:N
        Gj = genGrad(S.mdl, rho, S.p, S.massCfg, Phi, lam(idx(j)), idx(j));
        F(:,j,j) = Gj(:,1,1);
    end
    dOff = lam(idx) - lam(idx(1));
else
    dOff = [];
end
FJ = genGrad(S.mdl, rho, S.p, S.massCfg, Phi, lam(J), J);
fJJ = FJ(:,1,1);
Fraw = F; fJJraw = fJJ;
assert(strcmp(S.g('filter.type'),'sensitivity'), 'fp_state:filter', 'unexpected filter type');
filterAll = strcmp(S.g('filter.applyTo'),'all');
if filterAll
    for s = 1:N
        for k = s:N
            v = applyFilter(S.flt, rho, F(:,s,k));
            F(:,s,k) = v;  F(:,k,s) = v;
        end
    end
else
    for j = 1:N, F(:,j,j) = applyFilter(S.flt, rho, F(:,j,j)); end
end
fJJ = applyFilter(S.flt, rho, fJJ);

ctx = struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J), ...
             'rho',rho,'rhomin',S.rhomin,'volfrac',S.volfrac, ...
             'move',S.move, ...
             'maxInner',S.g('optimizer.inner.maxIterations'), ...
             'tolInner',S.g('optimizer.inner.tolerance'), ...
             'minInner',S.g('optimizer.inner.minIterations'), ...
             'offDiag',S.g('multiplicity.offDiagonal'), ...
             'dOff',dOff);

o.N = N; o.J = J; o.idx = idx;
o.omega_fresh_at_rho385 = w(1:S.Jcalc).';
o.lam_fresh_at_rho385   = lam(1:S.Jcalc).';
o.lam_cluster = ctx.lam(:).';  o.lamJ = ctx.lamJ; o.lamref = ctx.lam(1);
o.dOff = dOff(:).';  o.dOff_present = ~isempty(dOff);
o.offDiag = ctx.offDiag;
o.gap12_fresh = (w(2)-w(1))/w(1);
o.omega_hist_vs_fresh_maxrel = max(abs(o.omega_hist_at_rho385 - o.omega_fresh_at_rho385)./o.omega_fresh_at_rho385);
o.filter_guard_active_count = nnz(rho < 1e-3);   % max(1e-3,rho) guard
o.dOff_identity_check = struct( ...
    'lam2_minus_dOff2_minus_lam1', (ctx.lam(2) - dOff(2)) - ctx.lam(1), ...
    'tol_4eps_lam1', 4*eps*ctx.lam(1));

% ---- compare with the ctx retained by the prior audit -------------------
prior = fullfile(S.root,'diagnostics','filtered_subproblem_integrability_audit', ...
                 'evaluations','inner_kkt_state.mat');
o.prior_ctx_file = prior;
if isfile(prior)
    L = load(prior, 'ctx','stP','stC');
    c2 = L.ctx;
    rel = @(a,b) max(abs(a(:)-b(:)))/max(max(abs(b(:))),eps);
    o.prior_ctx_compare = struct( ...
        'F_maxrel', rel(ctx.F, c2.F), 'F_bitwise', isequal(ctx.F, c2.F), ...
        'fJJ_maxrel', rel(ctx.fJJ, c2.fJJ), 'fJJ_bitwise', isequal(ctx.fJJ, c2.fJJ), ...
        'lam_maxrel', rel(ctx.lam, c2.lam), 'lamJ_rel', abs(ctx.lamJ-c2.lamJ)/c2.lamJ, ...
        'dOff_maxabs', max(abs(ctx.dOff(:)-c2.dOff(:))), ...
        'rho_bitwise', isequal(ctx.rho, c2.rho), 'move_equal', ctx.move==c2.move);
    xP19  = L.stP.xFinal;  xM500 = L.stC.xFinal;
    o.P19_bitwise_equals_DRHO386 = isequal(xP19(1:S.NE), S.drho386);
    o.P19_nInner = L.stP.nInner;  o.M500_nInner = L.stC.nInner;
    o.P19_beta = L.stP.beta;      o.M500_beta = L.stC.beta;
    prior_relStep_P = L.stP.relHist(:).';  prior_relStep_C = L.stC.relHist(:).';
else
    o.prior_ctx_compare = 'prior ctx file absent';
    xP19 = [S.drho386; S.hist.beta(end)/ctx.lam(1)]; xM500 = [];
    prior_relStep_P = []; prior_relStep_C = [];
end

% ---- bounds -----------------------------------------------------------
NE = S.NE;
lo = max(ctx.rhomin - ctx.rho, -ctx.move);
hi = min(1          - ctx.rho,  ctx.move);
xmin = [lo; 0];  xmax = [hi; 5];
o.bounds = struct('n_lower_move_limited', nnz(-ctx.move > ctx.rhomin - ctx.rho), ...
    'n_lower_floor_limited', nnz(-ctx.move <= ctx.rhomin - ctx.rho), ...
    'n_upper_move_limited', nnz(ctx.move < 1 - ctx.rho), ...
    'n_upper_ceiling_limited', nnz(ctx.move >= 1 - ctx.rho), ...
    'min_width', min(hi-lo), 'max_width', max(hi-lo), 'bs_bounds', [0 5]);

% ---- verdict ------------------------------------------------------------
fail = {};
chk = @(nm,a,b) local_chk(nm,a,b);
fail = [fail, chk('rho386', o.rho386_sha256, expect.rho386)];
fail = [fail, chk('rho385', o.rho385_sha256, expect.rho385)];
fail = [fail, chk('drho386', o.drho386_sha256, expect.drho386)];
fail = [fail, chk('cfgHash', o.cfgHash, expect.cfgHash)];
fail = [fail, chk('implTree', o.implTree, expect.implTree)];
if ~o.implTree_ok,              fail{end+1} = 'source manifest verify'; end
if o.nOuter ~= expect.nOuter,   fail{end+1} = 'nOuter'; end
if o.stage  ~= expect.stage,    fail{end+1} = 'stage'; end
if o.move   ~= expect.move,     fail{end+1} = 'move'; end
if o.N      ~= expect.N,        fail{end+1} = 'N'; end
if o.multJ  ~= expect.multJ,    fail{end+1} = 'multJ'; end
if o.nInner_final ~= expect.nInner, fail{end+1} = 'nInner'; end
if isstruct(o.prior_ctx_compare)
    pc = o.prior_ctx_compare;
    if pc.F_maxrel > 1e-12 || pc.fJJ_maxrel > 1e-12 || pc.lam_maxrel > 1e-12 || ...
       pc.lamJ_rel > 1e-12 || pc.dOff_maxabs > 1e-12*ctx.lam(1) || ~pc.rho_bitwise || ~pc.move_equal
        fail{end+1} = 'retained ctx not reproduced';
    end
    if ~o.P19_bitwise_equals_DRHO386, fail{end+1} = 'P19 != DRHO(:,386)'; end
end
o.blockers = fail;
o.pass = isempty(fail);
if o.pass, o.verdict = 'FROZEN_480_PROBLEM25_STATE_PASS';
else,      o.verdict = 'FROZEN_480_PROBLEM25_STATE_FAIL'; end

% ---- persist the frozen context ---------------------------------------
if ~isfolder(ev), mkdir(ev); end
save(fullfile(ev,'frozen_ctx.mat'), 'ctx','xmin','xmax','xP19','xM500', ...
     'Fraw','fJJraw','prior_relStep_P','prior_relStep_C','-v7.3');
fid = fopen(fullfile(ev,'state_identity.json'),'w');
fprintf(fid,'%s',jsonencode(o,'PrettyPrint',true)); fclose(fid);

fprintf('[fp_state] %s\n', o.verdict);
fprintf('  rho386 %s\n  rho385 %s\n  cfg    %s\n  impl   %s\n', ...
    o.rho386_sha256, o.rho385_sha256, o.cfgHash, o.implTree);
fprintf('  outer=%d stage=%d move=%.3g N=%d J=%d multJ=%d nInner=%d\n', ...
    o.nOuter,o.stage,o.move,o.N,o.J,o.multJ,o.nInner_final);
fprintf('  lam = %s  lamJ = %.10g  dOff = %s\n', mat2str(o.lam_cluster,12), o.lamJ, mat2str(o.dOff,12));
if isstruct(o.prior_ctx_compare), disp(o.prior_ctx_compare); end
for k=1:numel(fail), fprintf('  BLOCKER: %s\n', fail{k}); end
out = o;
end

function c = local_chk(nm,a,b)
if strcmp(a,b), c = {}; else, c = {sprintf('%s: %s ~= %s', nm, a, b)}; end
end
