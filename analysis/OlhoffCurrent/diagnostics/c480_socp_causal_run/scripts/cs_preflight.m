function PF = cs_preflight(parts)
%CS_PREFLIGHT  Single-factor preflight P1, P3-P7 (AUDIT_PREREGISTRATION.md 2.3).
%   P2 (source-diff audit) is scripts/cs_diff_audit.py.  No C480 treatment
%   iteration is executed here.  P3 runs three outer iterations of the CONTROL
%   formulation (repeatedMMA) only to prove driver identity; P6 runs a TOY mesh.
if nargin < 1, parts = {'P1','P3','P4','P5','P6','P7'}; end
S = cs_setup();
guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);
pfFile = fullfile(S.study,'evaluations','preflight_matlab.json');
if isfile(pfFile), PF = jsondecode(fileread(pfFile)); else, PF = struct(); end
PF.matlab = version;

L = load(S.controlTraj, 'cfg', 'hist', 'RHO', 'DRHO');
cfg = L.cfg;

if any(strcmp(parts,'P1'))
    P1 = struct();
    P1.cfgHash_stored = olhoffcurrent_config_hash(cfg);
    [cf, ~] = cp_config(480, 60);
    P1.cfgHash_fresh = olhoffcurrent_config_hash(cf);
    P1.treatment_cfg_isequal_control_cfg = true;   % the treatment passes L.cfg itself
    P1.pass = strcmp(P1.cfgHash_stored, S.expect.cfgHash) && strcmp(P1.cfgHash_fresh, S.expect.cfgHash);
    PF.P1 = P1; save_pf(pfFile, PF);
end

if any(strcmp(parts,'P3'))
    treat = struct('innerSolver','repeatedMMA','stopAfter',3,'expectOmega1',L.hist.omega(:,1), ...
                   'checkpointEvery',0,'checkpointFile','','resumeFrom','','crossEvery',0,'progress',false);
    t = tic; res = cs_olhoffSolveSOCP(cfg, treat); P3 = struct('wall_s', toc(t));
    n = numel(res.hist.N); P3.n = n;
    rho = cfg.design.initial*ones(28800,1); R = zeros(28800,n); Dm = zeros(28800,n);
    for k = 1:n
        Dm(:,k) = res.diag.drho{k}; rho = min(1, max(cfg.design.minimum, rho + Dm(:,k))); R(:,k) = rho;
    end
    P3.drho_bitwise  = n == 3 && isequal(Dm, L.DRHO(:,1:3));
    P3.rho_bitwise   = n == 3 && isequal(R, L.RHO(:,1:3));
    P3.omega_bitwise = n == 3 && isequal(res.hist.omega(:,1:3), L.hist.omega(:,1:3));
    P3.beta_bitwise  = n == 3 && isequal(res.hist.beta(1:3), L.hist.beta(1:3));
    P3.nInner_bitwise= n == 3 && isequal(res.hist.nInner(1:3), L.hist.nInner(1:3));
    P3.ex_bitwise    = n == 3 && isequaln(res.hist.exAmp(1:3), L.hist.exAmp(1:3)) && ...
                       isequaln(res.hist.exCos(1:3), L.hist.exCos(1:3));
    P3.termination = res.termination; P3.status = res.status;
    P3.pass = P3.drho_bitwise && P3.rho_bitwise && P3.omega_bitwise && P3.beta_bitwise && ...
              P3.nInner_bitwise && P3.ex_bitwise && isempty(res.termination);
    PF.P3 = P3; save_pf(pfFile, PF);
    fprintf('[P3] pass=%d wall=%.1fs\n', P3.pass, P3.wall_s);
end

if any(strcmp(parts,'P4')) || any(strcmp(parts,'P5'))
    assert(strcmp(cs_filehash(S.frozenCtx), S.expect.frozenCtxSha), 'frozen ctx hash');
    assert(strcmp(cs_filehash(S.conicRef), S.expect.conicRefSha), 'conic reference hash');
    FC = load(S.frozenCtx, 'ctx'); ctx = FC.ctx;
    RF = load(S.conicRef, 'xRef'); xRef = RF.xRef;
    flags = struct('outer',386,'N',2,'n',1,'Jcalc',5,'useOff',true,'offDiag',true, ...
        'useProj',false,'useDensityFilter',false,'useSensFilter',true,'innerLP',false,'innerDesignVar',false);
    treat0 = struct('crossEvery',0,'progress',false);
end

if any(strcmp(parts,'P4'))
    t = tic; [drho, st, rec] = cs_socp_inner(ctx, flags, treat0); P4 = struct('wall_s', toc(t));
    NE = numel(ctx.rho);
    P4.accepted = rec.accepted; P4.termination = rec.termination; P4.reason = rec.reason;
    if rec.accepted
        z = drho; zo = xRef(1:NE); mv = ctx.move;
        P4.bs = rec.bs; P4.bsO = xRef(end); P4.abs_dbs = abs(rec.bs - xRef(end));
        P4.gainRecovery = (rec.bs - 1)/(xRef(end) - 1);
        P4.d2 = norm(z - zo)/norm(zo); P4.dinf = norm(z - zo, inf)/mv;
        P4.gap = rec.gap; P4.boxComp = rec.boxComp; P4.candidate = rec.candidate;
        P4.solver = rec.acceptedSolver; P4.attempts = numel(rec.attempts);
        P4.exitflags = [rec.attempts.exitflag]; P4.tSolve = rec.tSolve; P4.tCertificate = rec.tCertificate;
        P4.candidateGaps = rec.candidateGaps; P4.candidateNames = rec.candidateNames;
        P4.beta = st.beta;
        P4.nInterior = rec.nInterior; P4.nGrayFullMove = rec.nGrayFullMove; P4.nGray = rec.nGray;
        P4.pass = P4.abs_dbs <= 1e-8 && P4.gainRecovery >= 0.999 && P4.d2 <= 0.01 && P4.dinf <= 0.1;
    else
        P4.attemptReasons = {}; if isfield(rec,'attempts'), P4.attemptReasons = {rec.attempts.reason}; end
        P4.pass = false;
    end
    PF.P4 = P4; save_pf(pfFile, PF);
    fprintf('[P4] pass=%d\n', P4.pass); disp(P4);
end

if any(strcmp(parts,'P5'))
    P5 = struct();
    f3 = flags; f3.N = 3;
    [~,~,r] = cs_socp_inner(ctx, f3, treat0);
    P5.N3 = struct('rejected', ~r.accepted && strcmp(r.termination,'SOCP_UNSUPPORTED_CASE_HIT') && ~isfield(r,'attempts'), 'reason', r.reason);
    c2 = ctx; c2.dOff = [0; ctx.dOff(2)*1.001];
    [~,~,r] = cs_socp_inner(c2, flags, treat0);
    P5.inconsistentOffsets = struct('rejected', ~r.accepted && strcmp(r.termination,'SOCP_UNSUPPORTED_CASE_HIT') && ~isfield(r,'attempts'), 'reason', r.reason);
    c3 = ctx; c3.F(1,1,2) = c3.F(1,1,2)*(1 + 1e-12);
    [~,~,r] = cs_socp_inner(c3, flags, treat0);
    P5.asymmetricF = struct('rejected', ~r.accepted && strcmp(r.termination,'SOCP_UNSUPPORTED_CASE_HIT') && ~isfield(r,'attempts'), 'reason', r.reason);
    f4 = flags; f4.offDiag = false; c4 = ctx; c4.offDiag = false;
    [~,~,r] = cs_socp_inner(c4, f4, treat0);
    P5.offDiagFalse = struct('rejected', ~r.accepted && strcmp(r.termination,'SOCP_UNSUPPORTED_CASE_HIT') && ~isfield(r,'attempts'), 'reason', r.reason);
    c5 = ctx; c5.volFun = @(d) deal(0, zeros(size(d)));
    [~,~,r] = cs_socp_inner(c5, flags, treat0);
    P5.volFun = struct('rejected', ~r.accepted && strcmp(r.termination,'SOCP_UNSUPPORTED_CASE_HIT') && ~isfield(r,'attempts'), 'reason', r.reason);
    P = fp_problem(ctx);
    xi = xRef; xi(end) = xi(end) + 1e-6;               % infeasible: rows violated by ~1e-6
    Ci = cs_socp_certify(P, xi, []);
    P5.infeasibleX = struct('rejected', ~Ci.certified, 'reason', Ci.reason, 'maxRow', Ci.maxRow);
    xs = [zeros(P.NE,1); 1];                            % feasible, suboptimal by ~bsO-1
    Cs = cs_socp_certify(P, xs, []);
    P5.suboptimalX = struct('rejected', ~Cs.certified, 'reason', Cs.reason, 'bestGap', Cs.bestGap);
    Cr = cs_socp_certify(P, xRef, []);                  % positive control: the oracle itself
    P5.oracleCertifies = struct('certified', Cr.certified, 'candidate', Cr.candidate, 'bestGap', Cr.bestGap);
    P5.pass = P5.N3.rejected && P5.inconsistentOffsets.rejected && P5.asymmetricF.rejected && ...
        P5.offDiagFalse.rejected && P5.volFun.rejected && P5.infeasibleX.rejected && ...
        P5.suboptimalX.rejected && P5.oracleCertifies.certified;
    PF.P5 = P5; save_pf(pfFile, PF);
    fprintf('[P5] pass=%d\n', P5.pass); disp(P5);
end

if any(strcmp(parts,'P6'))
    toy = cfg;
    toy.domain.mesh.nelx = 96; toy.domain.mesh.nely = 12; toy.runtime.name = 'CS_TOY_96x12';
    toyDir = fullfile(S.study,'evaluations','toy_smoke');
    if ~isfolder(toyDir), mkdir(toyDir); end
    ck = fullfile(toyDir,'toy_checkpoint.mat'); if isfile(ck), delete(ck); end
    base = struct('innerSolver','exactSOCP','stopAfter',6,'expectOmega1',[],'checkpointEvery',0, ...
                  'checkpointFile','','resumeFrom','','crossEvery',1,'progress',true);
    t = tic; S6 = cs_olhoffSolveSOCP(toy, base); P6 = struct('wall_straight_s', toc(t));
    a1 = base; a1.stopAfter = 3; a1.checkpointEvery = 3; a1.checkpointFile = ck;
    A1 = cs_olhoffSolveSOCP(toy, a1); %#ok<NASGU>
    a2 = base; a2.resumeFrom = ck;
    A2 = cs_olhoffSolveSOCP(toy, a2);
    P6.straight_status = S6.status; P6.straight_n = numel(S6.hist.N);
    P6.all_accepted = all(cellfun(@(r) r.accepted, S6.socp(~cellfun(@isempty,S6.socp))));
    P6.nAccepted = sum(cellfun(@(r) r.accepted, S6.socp(~cellfun(@isempty,S6.socp))));
    P6.solvers = cellfun(@(r) r.acceptedSolver, S6.socp(~cellfun(@isempty,S6.socp)), 'UniformOutput', false);
    P6.resume_rho_bitwise = isequal(A2.rho, S6.rho);
    P6.resume_hist_bitwise = isequaln(A2.hist.omega, S6.hist.omega) && isequaln(A2.hist.beta, S6.hist.beta) ...
        && isequaln(A2.hist.exAmp, S6.hist.exAmp) && isequal(A2.hist.move, S6.hist.move);
    P6.resume_drho_bitwise = isequal(A2.diag.drho, S6.diag.drho);
    P6.resume_n = numel(A2.hist.N);
    meta = struct('note','TOY SMOKE TEST -- software verification only, excluded from all analyses');
    out = cs_postprocess(S6, toy, 'TOY96x12_smoke', toyDir, toyDir, meta);
    P6.postprocess_errors = {};
    for fn = {'socpTableError','cvExportError','supplementError'}
        if isfield(out, fn{1}), P6.postprocess_errors{end+1} = [fn{1} ': ' out.(fn{1})]; end
    end
    P6.postprocess_rebuildExact = out.rebuildExact;
    P6.pass = strcmp(P6.straight_status,'STOPPED_OTHER') && P6.straight_n == 6 && P6.all_accepted && ...
        P6.nAccepted == 6 && P6.resume_rho_bitwise && P6.resume_hist_bitwise && P6.resume_drho_bitwise && ...
        isempty(P6.postprocess_errors) && P6.postprocess_rebuildExact;
    PF.P6 = P6; save_pf(pfFile, PF);
    fprintf('[P6] pass=%d\n', P6.pass); disp(P6);
end

if any(strcmp(parts,'P7'))
    sm = olhoffcurrent_source_manifest('Verify', true);
    PF.P7 = struct('treeHash', sm.treeHash, 'ok', sm.ok, 'pass', sm.ok && strcmp(sm.treeHash, S.expect.implTree));
    save_pf(pfFile, PF);
end
end

function save_pf(f, PF)
cs_json(f, PF);
end
