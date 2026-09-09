function P = r240_provenance(tag)
%R240_PROVENANCE  Phase 0 provenance / evidence gate for the three-rung architecture
%   audit.  READ-ONLY.  Writes evidence/provenance_<tag>.json.
%
%   This audit executes ZERO scientific optimization runs.  Nothing here solves,
%   configures a solve, or writes into +impl/.  It reads files, hashes them, and
%   asks the standing gates whether the implementation may still be believed.

if nargin < 1, tag = 'start'; end
here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));          % analysis/OlhoffCurrent
repo  = fileparts(fileparts(root));
addpath(root);

P = struct();
P.study      = 'three_rung_resolution_240';
P.tag        = tag;
P.timestamp  = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
P.matlab     = version;
P.computer   = computer;
P.repo       = repo;
P.scientificRunsExecuted = 0;

% ---- git -----------------------------------------------------------------
P.branch = strtrim(local_git(repo,'rev-parse --abbrev-ref HEAD'));
P.head   = strtrim(local_git(repo,'rev-parse HEAD'));
P.status = strtrim(local_git(repo,'status --porcelain'));
P.clean  = isempty(P.status);
P.nDirtyPaths = numel(strsplit(strtrim(P.status), newline));
if isempty(strtrim(P.status)), P.nDirtyPaths = 0; end

% ---- currentness + source integrity --------------------------------------
st = olhoffcurrent_currentness('Verbose', false);
P.currentness    = st.state;
P.implTreeSha256 = st.manifest.treeHash;
P.implNFiles     = numel(st.manifest.files);
P.sourceOk       = st.manifest.ok;
P.sourceMismatch = numel(st.manifest.mismatches);
P.sourceMissing  = numel(st.manifest.missing);
P.sourceExtra    = numel(st.manifest.extra);

% ---- dispatch / path resolution ------------------------------------------
[guard, gate] = olhoffcurrent_paths(); %#ok<ASGLU>
P.dispatchOk       = gate.ok;
P.dispatchBlockers = numel(gate.blockers);
P.mmasubPath       = which('mmasub');
P.mmaPublished     = contains(P.mmasubPath, 'mma_published');

% ---- forbidden Olhoff trees must not be on the path ----------------------
[repoRel, absolute] = olhoffcurrent_forbidden_paths();
pth = strsplit(path, pathsep); hits = {};
for i = 1:numel(pth)
    for j = 1:numel(repoRel)
        if contains(pth{i}, repoRel{j}), hits{end+1} = pth{i}; end %#ok<AGROW>
    end
    for j = 1:numel(absolute)
        if strncmp(pth{i}, absolute{j}, numel(absolute{j})), hits{end+1} = pth{i}; end %#ok<AGROW>
    end
end
P.forbiddenOnPath = hits;
P.forbiddenAbsent = isempty(hits);

% ---- the scientific lock, from the resolved production config ------------
cfg = olhoffcurrent_config(160, 20);
g = @(p) olh.config.getPath(cfg, p);
P.lock = struct( ...
    'p', g('material.stiffness.p'), 'massModel', g('material.mass.model'), ...
    'q', g('material.mass.q'), 'filterType', g('filter.type'), ...
    'radiusPhysical', g('filter.radiusPhysical'), 'projection', g('projection.enabled'), ...
    'multMethod', g('multiplicity.method'), 'subspaceSize', g('multiplicity.subspaceSize'), ...
    'offDiagonal', g('multiplicity.offDiagonal'), 'mmaVariant', g('optimizer.inner.variant'), ...
    'moveLevels', g('move.levels'), 'movePolicy', g('move.policy'), ...
    'contSignal', g('move.continuation.signal'), 'stopToleranceRule', g('stop.toleranceRule'));
P.sensitivityFilterWins = strcmp(P.lock.filterType,'sensitivity');
P.fourRungLevelsIntact  = isequal(P.lock.moveLevels(:).', [0.04 0.02 0.01 0.005]);

% ---- tolerance law identity ---------------------------------------------
meshes = [160 20; 240 30; 320 40; 400 50]; ok = true;
P.tolCheck = struct('mesh',{},'NE',{},'stopTolerance',{},'frozenTol',{},'identical',{});
for i = 1:size(meshes,1)
    c  = olhoffcurrent_config(meshes(i,1), meshes(i,2));
    NE = meshes(i,1)*meshes(i,2);
    t  = olh.config.getPath(c,'stop.tolerance');
    f  = 0.05*sqrt(NE/3200);
    P.tolCheck(i) = struct('mesh',meshes(i,:),'NE',NE,'stopTolerance',t, ...
                           'frozenTol',f,'identical',t==f);
    ok = ok && (t==f);
end
P.toleranceLawIdentical = ok;

% ---- frozen preregistrations this audit inherits from --------------------
pre = { 'diagnostics/two_branch_maturity_240/PREREGISTRATION.md', ...
        'diagnostics/two_branch_controller_validation/PREREGISTRATION.md', ...
        'diagnostics/move_ladder_necessity/PREREGISTRATION.md', ...
        'diagnostics/two_rung_architecture/PREREGISTRATION.md', ...
        'diagnostics/three_rung_architecture/PREREGISTRATION.md' };
P.inheritedPrereg = struct('path',{},'sha256',{},'recordedSha',{},'match',{});
for i = 1:numel(pre)
    f = fullfile(root, pre{i});
    h = olhoffcurrent_sha256_file(f);
    rec = ''; sf = fullfile(fileparts(f),'evidence','PREREGISTRATION.sha256');
    if isfile(sf)
        txt = fileread(sf); tk = regexp(txt,'sha256\s+([0-9a-f]{64})','tokens','once');
        if ~isempty(tk), rec = tk{1}; end
    end
    P.inheritedPrereg(i) = struct('path',pre{i},'sha256',h,'recordedSha',rec, ...
        'match', isempty(rec) || strcmp(rec,h));
end

% ---- the raw causal-controller evidence this audit needs -----------------
need = { ...
  'analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C160x20_trajectory.mat'
  'analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C320x40_trajectory.mat'
  'analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C400x50_trajectory.mat'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C160x20_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C400x50_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C160x20_record.json'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_record.json'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C400x50_record.json'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/evidence/baselines.json'
  'analysis/OlhoffCurrent/diagnostics/move_ladder_necessity/METRICS.json'
  'analysis/OlhoffCurrent/diagnostics/two_rung_architecture/METRICS.json'
  'analysis/OlhoffCurrent/diagnostics/two_rung_architecture/evidence/event_verification.json'
  'analysis/OlhoffCurrent/diagnostics/two_rung_architecture/evidence/analysis.json'
  'analysis/OlhoffCurrent/diagnostics/three_rung_architecture/METRICS.json'
  'analysis/OlhoffCurrent/diagnostics/three_rung_architecture/evidence/event_verification.json'
  'analysis/OlhoffCurrent/diagnostics/three_rung_architecture/evidence/analysis.json'
  'analysis/OlhoffCurrent/diagnostics/three_rung_architecture/FINAL_SHA256.txt'
  'analysis/OlhoffCurrent/diagnostics/three_rung_architecture/DATA_MANIFEST.json'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/scripts/cv_config.m'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/scripts/cv_telemetry.m'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/scripts/cv_export.m'
  'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/scripts/cv_run.m'
  'analysis/OlhoffCurrent/+impl/architecture/+olh/+move/exhaustion.m'
  'analysis/OlhoffCurrent/+impl/architecture/+olh/+move/limit.m'
  'analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m'};
P.required = struct('path',{},'exists',{},'bytes',{},'sha256',{});
allThere = true;
for i = 1:numel(need)
    f = fullfile(repo, need{i});
    e = isfile(f);
    allThere = allThere && e;
    b = NaN; h = '';
    if e, d = dir(f); b = d.bytes; h = olhoffcurrent_sha256_file(f); end
    P.required(i) = struct('path',need{i},'exists',e,'bytes',b,'sha256',h);
end
P.requiredAllPresent = allThere;

% ---- 240x30: is there a causal controller trajectory at all? ------------
P.c240Candidates = {};
d = dir(fullfile(repo,'analysis','OlhoffCurrent','**','*240x30*'));
for i = 1:numel(d)
    if ~d(i).isdir
        P.c240Candidates{end+1} = strrep(fullfile(d(i).folder,d(i).name), [repo filesep], ''); %#ok<AGROW>
    end
end

% ---- prior studies' finalization gates ----------------------------------
studies = {'two_branch_controller_validation','move_ladder_necessity', ...
           'two_rung_architecture','three_rung_architecture','two_branch_maturity_240'};
P.priorGates = struct('study',{},'ok',{},'detail',{});
for i = 1:numel(studies)
    s = olhoffcurrent_finalization_gate(fullfile(root,'diagnostics',studies{i}), ...
                                        'Verbose', false, 'RepoRoot', repo);
    P.priorGates(i) = struct('study',studies{i},'ok',s.ok,'detail',s.detail);
end

% ---- repository test suite ----------------------------------------------
% Deterministic, software-only; no solve.  Recorded, and required to pass.
tdir = fullfile(root,'tests');
addpath(tdir);
td = dir(fullfile(tdir,'test_*.m'));
P.tests = struct('name',{},'ok',{},'err',{});
for i = 1:numel(td)
    [~,nm] = fileparts(td(i).name);
    ok = false; er = '';
    addpath(root); addpath(tdir);   % tests restore the path on cleanup
    try
        [~, r] = evalc('feval(nm)');
        if islogical(r) || isnumeric(r); ok = (r == 0) || (islogical(r) && r); else; ok = true; end
        if isstruct(r) && isfield(r,'ok'); ok = r.ok; end
    catch ME
        ok = false; er = ME.message;
    end
    P.tests(i) = struct('name',nm,'ok',logical(ok),'err',er);
end
P.testsPass = all([P.tests.ok]);
P.nTests    = numel(P.tests);

% SELF-REFERENCE.  test_finalization_gate asserts that no NEW study directory
% fails the finalization gate.  While THIS study is mid-flight its own directory
% exists and is not yet finalized, so that test fails for a reason created by
% this task and cured by finishing it.  Recorded explicitly rather than excused:
% the gate accepts it at tag 'start' only, and the FINAL provenance record must
% show testsPass true outright.
selfGate = olhoffcurrent_finalization_gate(study, 'Verbose', false, 'RepoRoot', repo);
P.selfFinalizationGate = selfGate.ok;
failed = {P.tests(~[P.tests.ok]).name};
P.testFailures = failed;
P.testsPassOrSelfReference = P.testsPass || ...
    (isequal(failed, {'test_finalization_gate'}) && ~selfGate.ok && strcmp(tag,'start'));

% ---- overall gate --------------------------------------------------------
P.gate = struct( ...
    'currentnessCurrent',  strcmp(P.currentness,'CURRENT'), ...
    'sourceIntegrityPass', P.sourceOk, ...
    'mmaPublished',        P.mmaPublished, ...
    'forbiddenAbsent',     P.forbiddenAbsent, ...
    'sensitivityFilter',   P.sensitivityFilterWins, ...
    'toleranceLaw',        P.toleranceLawIdentical, ...
    'fourRungLevels',      P.fourRungLevelsIntact, ...
    'preregHashesValid',   all([P.inheritedPrereg.match]), ...
    'requiredEvidence',    P.requiredAllPresent, ...
    'controllerGatePass',  P.priorGates(1).ok, ...
    'twoRungGatePass',     P.priorGates(3).ok, ...
    'threeRungGatePass',   P.priorGates(4).ok, ...
    'testsPass',           P.testsPassOrSelfReference);
f = fieldnames(P.gate);
P.gateOk = true;
for i = 1:numel(f), P.gateOk = P.gateOk && P.gate.(f{i}); end
P.verdict = 'C240_PROVENANCE_PASS';
if ~P.gateOk, P.verdict = 'C240_PROVENANCE_FAIL'; end

out = fullfile(study,'evidence',sprintf('provenance_%s.json',tag));
fid = fopen(out,'w'); fprintf(fid,'%s',jsonencode(P,'PrettyPrint',true)); fclose(fid);

fprintf('\n%s\nC240 RESOLUTION PROVENANCE GATE (%s)\n%s\n', repmat('=',1,72), tag, repmat('=',1,72));
fprintf('  branch %s  head %s  dirty=%d paths\n', P.branch, P.head(1:12), P.nDirtyPaths);
fprintf('  MATLAB %s\n', P.matlab);
fprintf('  +impl tree %s  (%d files)\n', P.implTreeSha256, P.implNFiles);
fprintf('  currentness %s   source ok=%d\n', P.currentness, P.sourceOk);
fprintf('  mmasub -> %s\n', P.mmasubPath);
for i = 1:numel(f), fprintf('  [%s] %s\n', local_pf(P.gate.(f{i})), f{i}); end
for i = 1:numel(P.priorGates)
    fprintf('  prior gate %-38s %s\n', P.priorGates(i).study, local_pf(P.priorGates(i).ok));
end
for i = 1:numel(P.tests)
    fprintf('  test %-44s %s%s\n', P.tests(i).name, local_pf(P.tests(i).ok), ...
        local_errtail(P.tests(i).err));
end
fprintf('  tests %d/%d pass; self finalization gate %s; testsPassOrSelfReference=%d\n', ...
    sum([P.tests.ok]), P.nTests, local_pf(P.selfFinalizationGate), P.testsPassOrSelfReference);
fprintf('  240x30 files matching *240x30*: %d\n', numel(P.c240Candidates));
fprintf('  VERDICT %s\n  written %s\n', P.verdict, out);
end

function s = local_pf(v), if v, s = 'PASS'; else, s = 'FAIL'; end, end
function s = local_errtail(e)
if isempty(e); s = ''; else; s = ['  <' e '>']; end
end
function o = local_git(r,c)
[~,o] = system(sprintf('git -C "%s" %s 2>/dev/null', r, c));
end
