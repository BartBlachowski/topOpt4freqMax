function test_a4_pipeline()
%TEST_A4_PIPELINE  Lightweight end-to-end A4 pipeline check.
%
%   Exercises the FULL A4 artifact pipeline on a TINY mesh (40x5, 4 iterations):
%     driver -> arms -> endpoint eval -> classifier -> result schema ->
%     topology CSV -> Table A4-1 -> figures -> manifest -> acceptance gate.
%
%   NO PRODUCTION OPTIMIZATION IS EXECUTED. This is a plumbing test, not science:
%   the numbers it produces are meaningless and are never used as evidence.
%   Gate A4-Pre is exercised separately (it is disabled for the sweep here, so
%   the sweep is tested even when the toy spectrum is inadmissible).

fprintf('\n=== test_a4_pipeline (tiny mesh; NOT production) ===\n');
nPass = 0; nFail = 0;

thisDir  = fileparts(mfilename('fullpath'));          % <repo>/scripts/revision_v1
repoRoot = fileparts(fileparts(thisDir));             % <repo>
rv1      = fullfile(repoRoot, 'examples', 'Revision_v1');
addpath(rv1);
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

outDir = fullfile(tempdir, sprintf('a4_pipeline_%s', datestr(now, 'HHMMSSFFF'))); %#ok<TNOW1,DATST>
mkdir(outDir);
cleanup = onCleanup(@() rmdir(outDir, 's'));

% ---- tiny base config, derived from the REAL base (single source) --------
base = jsondecode(fileread(fullfile(rv1, 'a4_ss_400x50_base.json')));
base.domain.mesh.nelx = 40;
base.domain.mesh.nely = 5;
base.optimization.max_iters = 4;
base.optimization.convergence_tol = 1e-16;  % land exactly on the cap (must be > 0)
base.optimization.volume_fraction = 0.9;    % near-solid: keeps a support-connected mode
tinyPath = fullfile(outDir, 'a4_tiny_base.json');
fid = fopen(tinyPath, 'w');
fprintf(fid, '%s\n', jsonencode(base, PrettyPrint=true));
fclose(fid);

opts = struct('base_config', tinyPath, 'n_levels', [Inf, 2], 'n_modes', 6, ...
              'run_pre_screen', false);

res = [];
try
    evalc('res = a4_eigenpair_refresh(outDir, opts);');
    [nPass, nFail] = ck('driver ran and returned a result struct', ...
        isstruct(res) && isfield(res, 'arms'), nPass, nFail);
catch ME
    [nPass, nFail] = ck(sprintf('driver ran (got: [%s] %s)', ME.identifier, ME.message), ...
        false, nPass, nFail);
    fprintf('\n  passed: %d   failed: %d\n', nPass, nFail);
    error('test_a4_pipeline:DriverFailed', 'A4 driver failed: %s', ME.message);
end

% ---- schema (spec §7.5) -------------------------------------------------
[nPass, nFail] = ck('one arm per declared N level', numel(res.arms) == 2, nPass, nFail);
armFields = {'N','base_config_hash','pmass','baseline','load_sensitivity', ...
    'omega1_tracked','mode_index_jstar','mac_to_phi0','omega1_min','omega1_thresholded', ...
    'omega1_omega2_gap','n_refresh','refresh_events','eigensolves_analytic', ...
    'iterations','cap','tol','final_design_change','grayness','feasibility', ...
    'wall_clock_s','class','breakdown','class_reason'};
[nPass, nFail] = ck('arm schema complete (spec §7.5)', ...
    all(isfield(res.arms(1), armFields)), nPass, nFail);
[nPass, nFail] = ck('every arm classified', ...
    all(arrayfun(@(a) ~isempty(a.class), res.arms)), nPass, nFail);

% CRITICAL: an arm that silently threw would be REJECTED and still be
% "classified". Without this assertion the whole pipeline test passes on a
% broken driver -- which is exactly how two real bugs (a4_endpoint_export not
% forwarded; convergence_tol=0 rejected) initially slipped through.
rejected = res.arms(arrayfun(@(a) strcmp(a.class, 'REJECTED'), res.arms));
[nPass, nFail] = ck(sprintf('NO arm REJECTED (machinery intact); %s', ...
    localRejectMsg(rejected)), isempty(rejected), nPass, nFail);

% The endpoint export must have produced real numbers, not NaN placeholders.
[nPass, nFail] = ck('endpoint eval produced finite true omega1 for every arm', ...
    all(isfinite([res.arms.omega1_tracked])), nPass, nFail);
[nPass, nFail] = ck('single-factor: all arms share one base-config hash', ...
    numel(unique({res.arms.base_config_hash})) == 1, nPass, nFail);
[nPass, nFail] = ck('decision emitted', ...
    isfield(res, 'decision') && ~isempty(res.decision.outcome), nPass, nFail);

% ---- the frozen arm must perform ZERO in-loop eigensolves (V-A4-4) ------
frozen = res.arms([res.arms.N] == Inf);
[nPass, nFail] = ck('V-A4-4: N=inf performs ZERO refreshes (one eigensolve at init)', ...
    frozen(1).n_refresh == 0, nPass, nFail);

refreshed = res.arms(~isinf([res.arms.N]));
if ~isempty(refreshed)
    r = refreshed(1);
    % Must ACTUALLY refresh (or be a legitimate B3). A vacuous 0==0 pass would
    % hide a dead R-1 path.
    okCount = (r.n_refresh == r.n_refresh_predicted) && ...
              (r.n_refresh > 0 || r.refresh_inadmissible);
    [nPass, nFail] = ck(sprintf(['V-A4-3: refreshed arm actually refreshed, count == ' ...
        'floor(nIter/N) (n_refresh=%d, predicted=%d)'], r.n_refresh, r.n_refresh_predicted), ...
        okCount, nPass, nFail);
end

% ---- artifacts ----------------------------------------------------------
req = {'a4_eigenpair_refresh_results.mat', 'a4_result.json', 'a4_manifest.json', 'a4_table.md'};
for i = 1:numel(req)
    p = fullfile(outDir, req{i});
    [nPass, nFail] = ck(sprintf('artifact present: %s', req{i}), ...
        isfile(p) && dir(p).bytes > 0, nPass, nFail);
end

% ---- JSON round-trips (schema validation) -------------------------------
ok = false;
try
    j = jsondecode(fileread(fullfile(outDir, 'a4_result.json')));
    ok = isfield(j, 'arms') && isfield(j, 'decision') && isfield(j, 'base_config_hash');
catch
end
[nPass, nFail] = ck('a4_result.json decodes and carries the schema', ok, nPass, nFail);

okm = false;
try
    m = jsondecode(fileread(fullfile(outDir, 'a4_manifest.json')));
    okm = isfield(m, 'stage') && strcmp(m.stage, 'A4') && isfield(m, 'files');
catch
end
[nPass, nFail] = ck('a4_manifest.json decodes; stage = A4', okm, nPass, nFail);

% ---- plotting pipeline --------------------------------------------------
figs = dir(fullfile(outDir, 'a4_fig*.png'));
[nPass, nFail] = ck(sprintf('plotting pipeline produced figures (%d)', numel(figs)), ...
    numel(figs) >= 3, nPass, nFail);

% ---- Gate A4-Pre is callable (one tiny checkpoint) ----------------------
okPre = false;
try
    evalc(['sr = a4_preflight_spectral_screen(outDir, ' ...
           'struct(''base_config'', tinyPath, ''checkpoints'', 2, ''n_modes'', 6, ''final_iters'', 4));']);
    okPre = true;
catch ME
    fprintf('    (pre-screen raised [%s])\n', ME.identifier);
end
[nPass, nFail] = ck('Gate A4-Pre executes and writes a verdict', ...
    okPre && isfile(fullfile(outDir, 'a4_pre_screen.json')), nPass, nFail);

% ==== CONVERGED-PATH REGRESSION (production-readiness audit, blocker FM1) ====
% The sweep above runs at max_iters=4 with tol=1e-16, so every arm CAPS and
% takes the B4-unattributed classifier branch.  That masked a production
% blocker: info.last_change was never exported by topopt_freq, so
% final_design_change was NaN for every arm and check_a4_run classified every
% CONVERGED arm as REJECTED -- i.e. the campaign's healthiest outcome failed
% its own acceptance gate.  This section exercises the CONVERGED branch:
% a near-solid 40x5 run with a reachable tolerance converges before the cap
% and MUST classify ACCEPTED (Class B).
convBase = base;
convBase.optimization.max_iters = 300;          % generous cap
convBase.optimization.convergence_tol = 1e-2;   % reachable before the cap
convPath = fullfile(outDir, 'a4_tiny_converged.json');
fid = fopen(convPath, 'w');
fprintf(fid, '%s\n', jsonencode(convBase, PrettyPrint=true));
fclose(fid);

convOpts = struct('base_config', convPath, 'n_levels', [Inf, 2], 'n_modes', 6, ...
                  'run_pre_screen', false);
convDir = fullfile(outDir, 'converged');
mkdir(convDir);
cres = [];
try
    evalc('cres = a4_eigenpair_refresh(convDir, convOpts);');
    [nPass, nFail] = ck('converged-path: driver ran', ...
        isstruct(cres) && isfield(cres, 'arms'), nPass, nFail);
catch ME
    [nPass, nFail] = ck(sprintf('converged-path: driver ran (got: [%s] %s)', ...
        ME.identifier, ME.message), false, nPass, nFail);
end

if isstruct(cres) && isfield(cres, 'arms')
    % (1) The runs must actually converge before the cap -- otherwise this
    % section is vacuously re-testing the capped branch.
    convergedAll = all(arrayfun(@(a) a.iterations < a.cap, cres.arms));
    [nPass, nFail] = ck(sprintf('converged-path: every arm stopped before the cap (%s)', ...
        strjoin(arrayfun(@(a) sprintf('%d/%d', a.iterations, a.cap), cres.arms, ...
        'UniformOutput', false)', ', ')), convergedAll, nPass, nFail);

    % (2) THE FM1 REGRESSION: final_design_change must be a real number that
    % reached the classifier (info.last_change export), not NaN.
    fdcOk = all(arrayfun(@(a) isfinite(a.final_design_change) && ...
        a.final_design_change <= a.tol, cres.arms));
    [nPass, nFail] = ck('converged-path: final_design_change finite and <= tol for every arm', ...
        fdcOk, nPass, nFail);

    % (3) A clean converged arm must classify ACCEPTED (Class B) -- before the
    % fix it classified REJECTED ("unconverged without a recognized breakdown
    % signature (change NaN > tol)").
    clsOk = all(arrayfun(@(a) strcmp(a.class, 'ACCEPTED'), cres.arms));
    [nPass, nFail] = ck(sprintf('converged-path: every arm Class B/ACCEPTED (%s)', ...
        strjoin(arrayfun(@(a) sprintf('N=%s:%s%s', a.tag, a.class, a.breakdown), ...
        cres.arms, 'UniformOutput', false)', ', ')), clsOk, nPass, nFail);

    % (4) With all arms Class B and gains within delta, the pre-registered
    % rule 1 (spec 5.3) must emit H0 -- not INDETERMINATE, and never the
    % previously-vacuous FROZEN_EXCEEDS_CLEAN_REFRESH.
    [nPass, nFail] = ck(sprintf('converged-path: decision is H0 via rule 1 (got %s)', ...
        cres.decision.outcome), ...
        strcmp(cres.decision.outcome, 'H0_FREEZING_IS_BENIGN'), nPass, nFail);
end

fprintf('\n  passed: %d   failed: %d\n', nPass, nFail);
if nFail > 0
    error('test_a4_pipeline:Failed', '%d A4 pipeline check(s) failed.', nFail);
end
fprintf('  A4 PIPELINE OK (tiny mesh; no production optimization executed)\n\n');
end

function m = localRejectMsg(rejected)
if isempty(rejected)
    m = 'none';
else
    m = sprintf('REJECTED: %s -> %s', rejected(1).tag, rejected(1).class_reason);
end
end

function [nPass, nFail] = ck(name, cond, nPass, nFail)
if cond
    fprintf('  [PASS] %s\n', name); nPass = nPass + 1;
else
    fprintf(2, '  [FAIL] %s\n', name); nFail = nFail + 1;
end
end
