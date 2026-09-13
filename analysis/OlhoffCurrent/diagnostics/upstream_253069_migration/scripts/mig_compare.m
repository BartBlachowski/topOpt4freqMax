function S = mig_compare(outJson)
%MIG_COMPARE  All 160x20 run comparisons of the migration (MIGRATION_PREREGISTRATION §4-§5).
%   Read-only over saved results.  Bitwise leaf comparison: numeric and logical
%   arrays by class, size and raw bytes; char/cell/struct recursively.  Excluded
%   everywhere: the nondeterministic timing fields tEig, tGrad, tInner, tOuter,
%   wallclock, and cfg.provenance.resolvedAt.
P = mig_paths();
mig_use_snapshot(P.up, true);                 % anchorRecord / anchorDigests / getPath
L = @(n) load(fullfile(P.ev, [n '.mat']));
EXC = {'tEig', 'tGrad', 'tInner', 'tOuter', 'wallclock', 'resolvedAt'};
S = struct();
S.standard = 'bitwise (class+size+raw bytes for numeric/logical; recursive for struct/cell); excluded: tEig tGrad tInner tOuter wallclock provenance.resolvedAt';

R = struct();
for n = {'PRE_BETA','PRE_EX3','POST_BETA','POST_EX3','POST_EX4','POST_PED','UP_BETA','UP_EX3','UP_PED'}
    R.(n{1}) = L(n{1});
end
schemaRows = olh.config.schema(); schemaRows = schemaRows(:,1);

% ---- per-run record ---------------------------------------------------------
S.runs = struct();
for n = fieldnames(R).'
    S.runs.(n{1}) = local_metrics(R.(n{1}).res, R.(n{1}).meta);
end

% ---- metric definition check against recorded values ------------------------
C160 = jsondecode(fileread(P.ref.C160record));
rx = R.POST_EX4.res.rho;
S.metricDefinitions = struct( ...
    'Mnd_pct', '100*mean(4*rho.*(1-rho))', ...
    'gray_fraction', 'mean(rho > 0.1 & rho < 0.9)  (upstream repro/run_repro.m greyFraction)', ...
    'mid_fraction', 'mean(rho > 0.4 & rho < 0.6)', ...
    'check_vs_C160x20_record', struct( ...
        'Mnd_pct', [100*mean(4*rx.*(1-rx)), C160.Mnd_final], ...
        'gray', [mean(rx > 0.1 & rx < 0.9), C160.gray_final], ...
        'mid_open_0p4_0p6', [mean(rx > 0.4 & rx < 0.6), C160.mid_final], ...
        'mid_closed_0p4_0p6', [mean(rx >= 0.4 & rx <= 0.6), C160.mid_final], ...
        'mid_open_0p25_0p75', [mean(rx > 0.25 & rx < 0.75), C160.mid_final]));

% ---- A. historical beta-stall -------------------------------------------------
S.A_beta = struct();
S.A_beta.post_vs_pre = local_cmp(R.POST_BETA.res, R.PRE_BETA.res, EXC, schemaRows, R.POST_BETA.cfg, R.PRE_BETA.cfg);
B = load(P.ref.benchmarkRecords);
rec = B.records(strcmp({B.records.method_key}, 'olhoff'));
j = find(arrayfun(@(s) isequal(s.mesh(:).', [160 20]), rec), 1);
ref = rec(j); got = R.POST_BETA.res;
S.A_beta.post_vs_campaign = struct( ...
    'reference', 'examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat (olhoff, 160x20)', ...
    'reference_sha256', local_fileSha(P.ref.benchmarkRecords), ...
    'rho_bitwise', local_bytesEqual(double(got.rho(:)), double(ref.x(:))), ...
    'omega1_bitwise', local_bytesEqual(double(got.omega(1)), double(ref.omega(1))), ...
    'volume_bitwise', local_bytesEqual(mean(double(got.rho(:))), mean(double(ref.x(:)))), ...
    'outer', [numel(got.hist.N), double(ref.counts.outer_iterations)], ...
    'inner', [sum(got.hist.nInner), double(ref.counts.inner_iterations_total)], ...
    'status', {{got.status, ref.status}}, ...
    'effective_config_hash_recorded', ref.effective_config_hash);
S.A_beta.post_vs_campaign.pass = S.A_beta.post_vs_campaign.rho_bitwise && S.A_beta.post_vs_campaign.omega1_bitwise && ...
    S.A_beta.post_vs_campaign.volume_bitwise && diff(S.A_beta.post_vs_campaign.outer) == 0 && ...
    diff(S.A_beta.post_vs_campaign.inner) == 0 && strcmp(got.status, 'CONVERGED') && strcmp(ref.status, 'NATIVE_CONVERGED');
S.A_beta.post_vs_up = local_cmp(R.POST_BETA.res, R.UP_BETA.res, EXC, schemaRows, R.POST_BETA.cfg, R.UP_BETA.cfg);

% ---- A'. historical three-rung stage exhaustion ------------------------------
S.A_ex3 = struct();
S.A_ex3.post_vs_pre = local_cmp(R.POST_EX3.res, R.PRE_EX3.res, EXC, schemaRows, R.POST_EX3.cfg, R.PRE_EX3.cfg);
T = load(P.ref.targetEX3, 'r', 'cfg'); T.r.cfg = T.cfg;
S.A_ex3.post_vs_targetAudit = local_cmp(local_sub(R.POST_EX3.res, T.r), local_subref(T.r), EXC, {}, [], []);
S.A_ex3.post_vs_targetAudit.reference = 'Olhoff-upstream-capabilities-evidence/runs/case_TARGET_EX3_160.mat (OlhoffCurrent tree edbfe47, 2026-09-13)';
S.A_ex3.post_vs_targetAudit.reference_sha256 = local_fileSha(P.ref.targetEX3);
S.A_ex3.post_vs_up = local_cmp(R.POST_EX3.res, R.UP_EX3.res, EXC, schemaRows, R.POST_EX3.cfg, R.UP_EX3.cfg);
S.A_ex3.events = local_exEvents(R.POST_EX3.res);

% ---- A''. four-rung stage exhaustion vs the committed target record ----------
got = R.POST_EX4.res;
TR = load(P.ref.C160traj, 'RHO', 'DRHO', 'hist', 'exh', 'log', 'meta');
csv = readtable(P.ref.C160csv);
histNames = setdiff(fieldnames(TR.hist), EXC);
hd = {};
for k = 1:numel(histNames)
    if ~isfield(got.hist, histNames{k}) || ~local_leafEqual(got.hist.(histNames{k}), TR.hist.(histNames{k}))
        hd{end+1} = histNames{k}; %#ok<AGROW>
    end
end
drhoEq = isfield(got, 'diag') && size(TR.DRHO, 2) == numel(got.diag.drho) && ...
    all(arrayfun(@(k) local_bytesEqual(double(got.diag.drho{k}(:)), TR.DRHO(:,k)), 1:size(TR.DRHO,2)));
csvCols = {'omega1', 'move', 'stage', 'beta', 'nInner', 'cumInner', 'multN', 'exA', 'exB', 'exE', 'exDecl', 'exAmp'};
csvSrc  = {got.hist.omega(1,:), got.hist.move, got.hist.stage, got.hist.beta, got.hist.nInner, ...
           got.hist.cumInner, got.hist.N, got.hist.exA, got.hist.exB, got.hist.exE, got.hist.exDecl, got.hist.exAmp};
csvEq = struct();
for k = 1:numel(csvCols)
    csvEq.(csvCols{k}) = isequaln(double(csvSrc{k}(:)), double(csv.(csvCols{k})(:)));
end
ex = got.exhaustion;
S.A_ex4 = struct( ...
    'reference', 'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C160x20_record.json + _iterations.csv + evidence/two_branch_controller_validation/C160x20_trajectory.mat', ...
    'trajectory_sha256', local_fileSha(P.ref.C160traj), ...
    'rho_sha256_bytes', {{local_sha(typecast(double(got.rho(:)), 'uint8')), C160.rho_sha256}}, ...
    'rho_equals_trajectory_last_column', local_bytesEqual(double(got.rho(:)), TR.RHO(:,end)), ...
    'omega5_exact', isequal(double(got.omega(1:5)).', C160.omega(:).'), ...
    'nOuter', [numel(got.hist.N), C160.nOuter], 'innerTotal', [sum(got.hist.nInner), C160.innerTotal], ...
    'innerMax', [max(got.hist.nInner), C160.innerMax], ...
    'volume_exact', isequal(mean(got.rho), C160.volume_final), ...
    'status', {{got.status, C160.status}}, ...
    'hist_fields_vs_trajectory_differing', {hd}, 'hist_fields_compared', numel(histNames), ...
    'drho_all_iterations_bitwise', drhoEq, ...
    'csv_columns_exact', csvEq, ...
    'exhaustion_equal', isequaln(ex.stageStarts(:), TR.exh.stageStarts(:)) && isequaln(ex.descents, TR.exh.descents) && ...
        isequaln(ex.events, TR.exh.events) && isequal(ex.eventBranch, TR.exh.eventBranch) && ...
        isequal(ex.terminalDeclIter, TR.exh.terminalDeclIter) && isequal(ex.terminalBranch, TR.exh.terminalBranch), ...
    'exhaustion_record_isequaln', isequaln(ex, TR.exh), ...
    'log_equal', isequal(got.log(:), TR.log(:)), ...
    'record_log_equal', isequal(got.log(:), C160.log(:)));
S.A_ex4.pass = strcmp(S.A_ex4.rho_sha256_bytes{1}, S.A_ex4.rho_sha256_bytes{2}) && S.A_ex4.rho_equals_trajectory_last_column && ...
    S.A_ex4.omega5_exact && diff(S.A_ex4.nOuter) == 0 && diff(S.A_ex4.innerTotal) == 0 && diff(S.A_ex4.innerMax) == 0 && ...
    S.A_ex4.volume_exact && isempty(hd) && drhoEq && all(struct2array(csvEq)) && S.A_ex4.exhaustion_equal && ...
    S.A_ex4.log_equal && S.A_ex4.record_log_equal && strcmp(got.status, C160.status);
S.A_ex4.events = local_exEvents(got);

% ---- B. Pedersen / adaptive box -----------------------------------------------
Cm = load(P.ref.S160, 'res');
S.B_ped = struct();
S.B_ped.committed = struct('artifact', 'Olhoff repro/results/S160x20/res.mat', 'sha256', local_fileSha(P.ref.S160));
S.B_ped.post_vs_committed = local_cmp(R.POST_PED.res, Cm.res, EXC, schemaRows, R.POST_PED.cfg, Cm.res.cfg);
S.B_ped.post_vs_up = local_cmp(R.POST_PED.res, R.UP_PED.res, EXC, schemaRows, R.POST_PED.cfg, R.UP_PED.cfg);
h = R.POST_PED.res.hist; a = R.POST_PED.res.aux;
S.B_ped.box = struct('move_max_first5', h.move(1:5), 'move_max_last', h.move(end), ...
    'moveMean_first5', a.moveMean(1:5), 'moveMean_last', a.moveMean(end), ...
    'move_max_equals_committed', local_leafEqual(h.move, Cm.res.hist.move), ...
    'moveMean_equals_committed', local_leafEqual(a.moveMean, Cm.res.aux.moveMean), ...
    'Mnd_equals_committed', local_leafEqual(a.Mnd, Cm.res.aux.Mnd));

% ---- p-continuation anchors (known pre-existing defect) ------------------------
S.pcont = struct();
for lab = {'A6_pdecoupled160', 'A7_massp160'}
    tag = ['POST_' lab{1}(1:2)];
    X = L(tag);
    rec = anchorRecord(X.res, X.cfg); dig = anchorDigests(rec);
    Cd = load(P.ref.anchorCand(lab{1}), 'dig', 'rec');
    Rf = load(fullfile(P.ref.anchorsRef, [lab{1} '.mat']), 'rec');
    dr = anchorDigests(Rf.rec);
    hf = Rf.rec.histFields; diffHist = {};
    for k = 1:numel(hf)
        if ~isequaln(Rf.rec.hist.(hf{k}), rec.hist.(hf{k})); diffHist{end+1} = hf{k}; end %#ok<AGROW>
    end
    mv = find(Rf.rec.hist.move(:) ~= rec.hist.move(:)).';
    S.pcont.(lab{1}(1:2)) = struct('label', lab{1}, 'nOuter', rec.nOuter, 'status', rec.status, ...
        'science_digest_target', dig.science, 'science_digest_upstream_candidate', Cd.dig.science, ...
        'target_equals_upstream_candidate', strcmp(dig.science, Cd.dig.science) && strcmp(dig.logShape, Cd.dig.logShape) ...
            && strcmp(dig.presentation, Cd.dig.presentation), ...
        'target_equals_committed_reference', strcmp(dig.science, dr.science), ...
        'rho_equals_reference', isequal(rec.rho, Rf.rec.rho), 'omega_equals_reference', isequal(rec.omega, Rf.rec.omega), ...
        'drho_equals_reference', isequal(rec.drho, Rf.rec.drho), ...
        'hist_fields_differing_from_reference', {diffHist}, ...
        'move_history_differs_at', mv, 'pEvents', rec.transitions.pEvents, ...
        'move_differences_only_at_pEvents', ~isempty(mv) && all(ismember(mv, rec.transitions.pEvents)));
end

% ---- verdict inputs ------------------------------------------------------------
S.pass = struct( ...
    'beta_post_vs_pre', S.A_beta.post_vs_pre.passPrePost, ...
    'beta_post_vs_campaign', S.A_beta.post_vs_campaign.pass, ...
    'beta_post_vs_up', S.A_beta.post_vs_up.passStrict, ...
    'ex3_post_vs_pre', S.A_ex3.post_vs_pre.passPrePost, ...
    'ex3_post_vs_targetAudit', S.A_ex3.post_vs_targetAudit.passDiffs, ...
    'ex3_post_vs_up', S.A_ex3.post_vs_up.passStrict, ...
    'ex4_post_vs_committed_record', S.A_ex4.pass, ...
    'ped_post_vs_committed', S.B_ped.post_vs_committed.passCommitted, ...
    'ped_post_vs_up', S.B_ped.post_vs_up.passStrict, ...
    'pcont_A6_equals_upstream', S.pcont.A6.target_equals_upstream_candidate && S.pcont.A6.rho_equals_reference && ...
        S.pcont.A6.omega_equals_reference && S.pcont.A6.drho_equals_reference && S.pcont.A6.move_differences_only_at_pEvents, ...
    'pcont_A7_equals_upstream', S.pcont.A7.target_equals_upstream_candidate && S.pcont.A7.rho_equals_reference && ...
        S.pcont.A7.omega_equals_reference && S.pcont.A7.drho_equals_reference && S.pcont.A7.move_differences_only_at_pEvents);
fid = fopen(outJson, 'w'); fwrite(fid, jsonencode(S, 'PrettyPrint', true)); fclose(fid);
disp(S.pass);
end

% =============================================================================
function c = local_cmp(a, b, EXC, rows, cfgA, cfgB)
%LOCAL_CMP  a = candidate (migrated), b = reference.
d = local_diff(a, b, 'res', [EXC, {'cfg'}]);
onlyA = d(endsWith(d, '(only in candidate)'));
onlyB = d(endsWith(d, '(only in reference)'));
vals  = setdiff(d, [onlyA, onlyB], 'stable');
c = struct('differing', {vals}, 'onlyInCandidate', {onlyA}, 'onlyInReference', {onlyB});
% configuration over schema rows
if ~isempty(rows)
    cd = {};
    for r = 1:numel(rows)
        try, va = olh.config.getPath(cfgA, rows{r}); catch, va = '<absent>'; end
        try, vb = olh.config.getPath(cfgB, rows{r}); catch, vb = '<absent>'; end
        if ~local_leafEqual(va, vb); cd{end+1} = rows{r}; end %#ok<AGROW>
    end
    c.cfgRowsDiffering = cd;
else
    c.cfgRowsDiffering = {};
end
c.passDiffs  = isempty(vals);
c.passStrict = isempty(vals) && isempty(onlyA) && isempty(onlyB) && isempty(c.cfgRowsDiffering);
% pre -> post (MIGRATION_PREREGISTRATION §5/§6): the old target created the 12
% ex* trace fields always, EMPTY when the controller is off; upstream adds
% res.aux (reporting); the configuration may differ only in the six rows the
% schema gained (their values are checked neutral in config_transition.json).
exNames = regexp(onlyB, '^res\.hist\.(ex[A-Za-z]+) \(only in reference\)$', 'tokens', 'once');
exOnly = all(~cellfun(@isempty, exNames));
exEmpty = exOnly && all(cellfun(@(t) isempty(b.hist.(t{1})), exNames));
c.droppedFieldsAllEmptyExTrace = exEmpty;
addedRows = {'material.stiffness.linearBelow', 'optimizer.inner.asymptoteHistory', 'move.adaptive.grow', ...
             'move.adaptive.shrink', 'stop.guards.settledWindow', 'stop.guards.boxInactiveFraction'};
c.cfgRowsDifferingOnlyAddedRows = all(ismember(c.cfgRowsDiffering, addedRows));
c.passPrePost = isempty(vals) && all(ismember(onlyA, {'res.aux (only in candidate)'})) && ...
    (isempty(onlyB) || exEmpty) && c.cfgRowsDifferingOnlyAddedRows;
% committed 6b08708 result -> migrated: tOuter excluded; only stop.rule (absent at 6b08708),
% runtime.verbose and runtime.name may differ in configuration
c.passCommitted = isempty(vals) && isempty(onlyB) && all(ismember(c.cfgRowsDiffering, {'stop.rule', 'runtime.verbose', 'runtime.name'}));
end

function d = local_diff(a, b, path, EXC)
d = {};
if isstruct(a) && isstruct(b) && isscalar(a) && isscalar(b)
    f = union(fieldnames(a), fieldnames(b), 'stable');
    for k = 1:numel(f)
        if any(strcmp(f{k}, EXC)); continue; end
        p = [path '.' f{k}];
        if ~isfield(a, f{k}),     d{end+1} = [p ' (only in reference)']; %#ok<AGROW>
        elseif ~isfield(b, f{k}), d{end+1} = [p ' (only in candidate)']; %#ok<AGROW>
        else, d = [d, local_diff(a.(f{k}), b.(f{k}), p, EXC)]; %#ok<AGROW>
        end
    end
elseif iscell(a) && iscell(b) && isequal(size(a), size(b))
    for k = 1:numel(a)
        d = [d, local_diff(a{k}, b{k}, sprintf('%s{%d}', path, k), EXC)]; %#ok<AGROW>
    end
elseif ~local_leafEqual(a, b)
    d{end+1} = path;
end
end

function tf = local_leafEqual(a, b)
if (isnumeric(a) || islogical(a)) && (isnumeric(b) || islogical(b))
    tf = strcmp(class(a), class(b)) && isequal(size(a), size(b)) && ...
         (isempty(a) || isequal(typecast(local_bytes(a), 'uint8'), typecast(local_bytes(b), 'uint8')));
else
    tf = isequaln(a, b);
end
end

function u = local_bytes(v)
if islogical(v), u = uint8(v(:)); elseif ~isreal(v), u = typecast([real(v(:)); imag(v(:))], 'uint8');
else, u = typecast(v(:), 'uint8'); end
end

function tf = local_bytesEqual(a, b)
tf = isequal(size(a), size(b)) && isequal(typecast(a(:), 'uint8'), typecast(b(:), 'uint8'));
end

function m = local_metrics(res, meta)
r = double(res.rho(:)); w = double(res.omega(:));
m = struct('side', meta.side, 'case', meta.case, 'treeHash', meta.treeHash, 'solver', meta.solver, ...
    'status', res.status, 'nOuter', numel(res.hist.N), 'innerTotal', sum(res.hist.nInner), ...
    'innerMax', max(res.hist.nInner), 'innerNotConverged', sum(~res.hist.innerConv), ...
    'omega1', w(1), 'omega2', w(2), 'omega3', w(3), 'gap12_pct', 100*(w(2)-w(1))/w(1), ...
    'volume', mean(r), 'Mnd_pct', 100*mean(4*r.*(1-r)), 'gray_fraction', mean(r > 0.1 & r < 0.9), ...
    'mid_fraction', mean(r > 0.4 & r < 0.6), 'rho_sha256_bytes', local_sha(typecast(r, 'uint8')), ...
    'final_move_max', res.hist.move(end), 'final_stage', res.hist.stage(end), ...
    'final_dxNorm2', res.hist.dxNorm2(end), 'final_dxOuter', res.hist.dxOuter(end), ...
    'stage_starts', [1, find(diff(double(res.hist.stage(:).')) ~= 0) + 1], ...
    'move_levels_at_starts', [], 'final_N', res.hist.N(end), 'log', {res.log(:).'}, ...
    'wall_s_nondeterministic', meta.wall_s);
m.move_levels_at_starts = res.hist.move(m.stage_starts);
if isfield(res, 'aux') && isfield(res.aux, 'moveMean'), m.final_move_mean = res.aux.moveMean(end); end
if isfield(res, 'exhaustion'), m.exhaustion = local_exEvents(res); end
end

function e = local_exEvents(res)
x = res.exhaustion;
e = struct('stageStarts', x.stageStarts(:).', 'descents', x.descents, 'eventBranch', {x.eventBranch}, ...
    'terminalDeclared', x.terminalDeclared, 'terminalDeclIter', x.terminalDeclIter, ...
    'terminalDeclBegin', x.terminalDeclBegin, 'terminalBranch', x.terminalBranch);
end

function s = local_sub(res, ref)
%LOCAL_SUB  the fields of res that the upstream-audit record r carries
s = struct('rho', res.rho, 'omega', res.omega, 'lambda', res.lambda, 'hist', res.hist, 'log', {res.log}, ...
    'status', res.status, 'nOuter', res.nOuter, 'exhaustion', res.exhaustion, 'drho', {res.diag.drho});
if isfield(ref, 'aux'), s.aux = res.aux; end
end

function r = local_subref(ref)
r = rmfield(ref, intersect(fieldnames(ref), {'solver', 'wallclock', 'cfg'}));
end

function h = local_fileSha(p)
fid = fopen(p, 'r'); b = fread(fid, Inf, '*uint8'); fclose(fid);
h = local_sha(b);
end

function h = local_sha(bytes)
md = java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(bytes), md.update(bytes(:)); end
x = typecast(md.digest(), 'uint8'); h = lower(reshape(dec2hex(x, 2).', 1, []));
end
