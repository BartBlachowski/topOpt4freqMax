function o = cs_control_identity()
%CS_CONTROL_IDENTITY  Part 1: recover and verify the authoritative retained
%   C480 three-rung canary.  READ-ONLY.  No FE solve, no optimization.
S = cs_setup();
guard = olhoffcurrent_paths(); %#ok<NASGU>
o = struct('study','c480_socp_causal_run','part','control identity');
o.trajectory = S.controlTraj;

L = load(S.controlTraj, 'RHO','DRHO','move','hist','cfg','meta','exh','log');
rec = jsondecode(fileread(S.controlRecord));
[NE, n] = size(L.RHO);
o.NE = NE; o.nOuter = n;
rho0 = L.cfg.design.initial*ones(NE,1);
o.rho0_sha256   = cs_hash(rho0);
o.rho385_sha256 = cs_hash(L.RHO(:,385));
o.rho386_sha256 = cs_hash(L.RHO(:,386));
o.record_rho_sha256 = rec.rho_sha256;

% ---- config identity: stored, and freshly resolved -----------------------
o.cfgHash_stored = olhoffcurrent_config_hash(L.cfg);
[cfgFresh, ~] = cp_config(480, 60);
o.cfgHash_fresh = olhoffcurrent_config_hash(cfgFresh);
o.meta_cfgHash = L.meta.cfgHash;
o.meta_implTree = L.meta.implTree;
sm = olhoffcurrent_source_manifest('Verify', true);
o.implTree_now = sm.treeHash; o.implTree_verify_ok = sm.ok;

g = @(p) olh.config.getPath(L.cfg, p);
o.mesh = [g('domain.mesh.nelx') g('domain.mesh.nely')];
o.levels = g('move.levels');
o.signal = g('move.continuation.signal');
o.stopRule = g('stop.rule');
o.innerType = g('optimizer.inner.type');
o.innerVariable = g('optimizer.inner.variable');
o.innerVariant = g('optimizer.inner.variant');
o.multMethod = g('multiplicity.method'); o.subspaceSize = g('multiplicity.subspaceSize');
o.diagonalOffsets = g('multiplicity.diagonalOffsets'); o.offDiagonal = g('multiplicity.offDiagonal');
o.filterType = g('filter.type'); o.filterRadius = g('filter.radiusPhysical'); o.filterApplyTo = g('filter.applyTo');
o.projection = g('projection.enabled'); o.p = g('material.stiffness.p');
o.massModel = g('material.mass.model'); o.q = g('material.mass.q');
o.rhomin = g('design.minimum'); o.volfrac = g('design.volumeFraction');
o.eigenSolver = g('eigen.solver'); o.maxOuter = g('runtime.maxOuter');
o.stopTol = g('stop.tolerance');

% ---- trajectory structure -----------------------------------------------
h = L.hist;
o.hist_columns = numel(h.N);
st = h.stage(:);
o.stage_lengths = [nnz(st==1) nnz(st==2) nnz(st==3)];
o.moves_by_stage = [unique(h.move(st==1)) unique(h.move(st==2)) unique(h.move(st==3))];
o.stageStarts = L.exh.stageStarts(:).';
o.descents = L.exh.descents;
o.eventBranch = L.exh.eventBranch;
o.terminalDeclared = L.exh.terminalDeclared;
o.terminalDeclIter = L.exh.terminalDeclIter;
o.terminalDeclBegin = L.exh.terminalDeclBegin;
o.terminalBranch = L.exh.terminalBranch;
o.signalDrivesMove = L.exh.signalDrivesMove; o.ruleAdmitsStop = L.exh.ruleAdmitsStop;
decl = find(h.exDecl(:) > 0 & [true; diff(h.exDecl(:)) > 0]);
o.hist_declaration_iterations = decl(:).';
ev = struct('iter',{},'stage',{},'move',{},'branch',{},'amp',{},'amp_over_eps',{},'medcos',{},'mednet',{},'nA',{},'nB',{},'omega1',{},'terminal',{});
for i = 1:numel(decl)
    k = decl(i);
    br = 'A'; if h.exNB(k) >= 20, br = 'B'; end
    ev(end+1) = struct('iter',k,'stage',h.stage(k),'move',h.move(k),'branch',br, ...
        'amp',h.exAmp(k),'amp_over_eps',h.exAmp(k)/o.stopTol,'medcos',h.exMedcos(k), ...
        'mednet',h.exMednet(k),'nA',h.exNA(k),'nB',h.exNB(k),'omega1',h.omega(1,k), ...
        'terminal', k == n); %#ok<AGROW>
end
o.controller_events = ev;
o.innerTotal = sum(h.nInner); o.innerMax = max(h.nInner); o.innerNonConv = sum(~h.innerConv);
o.N_unique = unique(h.N); o.multJ_count = sum(h.multJ); o.multJ_iters = find(h.multJ);
o.beta_final = h.beta(end);
o.status_record = rec.status;
o.omega_record = rec.omega(:).';
o.omega1_record = rec.omega1; o.omega2_record = rec.omega2;
o.gap12_record = rec.gap12; o.gap23_record = rec.gap23;
o.volume_record = rec.volume_final;
r = L.RHO(:,end);
o.Mnd = 100*mean(4*r.*(1-r)); o.gray = mean(r>0.1 & r<0.9); o.mid = mean(r>=0.4 & r<=0.6);
o.volume = mean(r);
o.Mnd_record = rec.Mnd_final; o.gray_record = rec.gray_final; o.mid_record = rec.mid_final;
o.wall_record = rec.wall_s; o.t_record = rec.t;
o.sum_tEig = sum(h.tEig); o.sum_tGrad = sum(h.tGrad); o.sum_tInner = sum(h.tInner); o.sum_tOuter = sum(h.tOuter);
o.hist_omega_first = h.omega(:,1).';

% ---- rebuild check (as cp_run proved) -------------------------------------
rr = rho0; mx = 0;
for k = 1:n
    rr = min(1, max(o.rhomin, rr + L.DRHO(:,k)));
    mx = max(mx, max(abs(rr - L.RHO(:,k))));
end
o.rebuild_max_abs = mx;

% ---- checks -------------------------------------------------------------
E = S.expect; c = struct();
c.rho0 = strcmp(o.rho0_sha256, E.rho0Sha);
c.rho386 = strcmp(o.rho386_sha256, E.rho386Sha) && strcmp(rec.rho_sha256, E.rho386Sha);
c.cfg_stored = strcmp(o.cfgHash_stored, E.cfgHash);
c.cfg_fresh = strcmp(o.cfgHash_fresh, E.cfgHash);
c.meta = strcmp(o.meta_cfgHash, E.cfgHash) && strcmp(o.meta_implTree, E.implTree);
c.implTree_now = sm.ok && strcmp(sm.treeHash, E.implTree);
c.mesh = isequal(o.mesh, [480 60]);
c.levels = isequal(o.levels, [0.04 0.02 0.01]);
c.three_rung_policy = strcmp(o.signal,'stageExhaustion') && strcmp(o.stopRule,'stageExhaustion');
c.hist_columns = o.hist_columns == 386 && n == 386;
c.stage_lengths = isequal(o.stage_lengths, [308 39 39]) && isequal(o.moves_by_stage, [0.04 0.02 0.01]);
c.events = isequal(o.stageStarts, [1 309 348]) && isequal(o.hist_declaration_iterations, [308 347 386]) ...
    && o.terminalDeclared && o.terminalDeclIter == 386 && strcmp(o.terminalBranch,'B') ...
    && isequal(o.eventBranch(:).', {'B','B'}) && all(strcmp({ev.branch},'B'));
c.status = strcmp(rec.status,'CONVERGED');
c.metrics = abs(o.Mnd - rec.Mnd_final) <= 1e-12*rec.Mnd_final && ...
            abs(o.gray - rec.gray_final) <= 1e-12 && abs(o.mid - rec.mid_final) <= 1e-12 && ...
            abs(o.volume - rec.volume_final) <= 1e-12;
c.inner = o.innerTotal == 7300;
c.rebuild = o.rebuild_max_abs == 0;
c.N2 = isequal(o.N_unique, 2);
o.checks = c;
o.all_matlab_checks_pass = all(cellfun(@(f) logical(c.(f)), fieldnames(c)));
cs_json(fullfile(S.study,'evaluations','control_identity_matlab.json'), o);
fprintf('[cs_control_identity] all MATLAB checks pass = %d\n', o.all_matlab_checks_pass);
disp(c);
end
