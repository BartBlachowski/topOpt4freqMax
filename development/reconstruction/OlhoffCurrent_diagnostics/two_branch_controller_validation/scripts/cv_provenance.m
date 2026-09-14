function P = cv_provenance(tag)
%CV_PROVENANCE  Phase 0 provenance / currentness gate for the causal controller
%   validation study.  Read-only.  Writes evidence/provenance_<tag>.json.
%
%   Requires: currentness CURRENT, source integrity PASS, published MMA wins,
%   sensitivity filter wins, forbidden Olhoff paths absent, tests PASS.

if nargin < 1, tag = 'start'; end
here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));          % analysis/OlhoffCurrent
repo  = fileparts(fileparts(root));
addpath(root);

P = struct();
P.tag        = tag;
P.timestamp  = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
P.matlab     = version;
P.computer   = computer;
P.repo       = repo;

% ---- git -----------------------------------------------------------------
P.branch = strtrim(local_git(repo,'rev-parse --abbrev-ref HEAD'));
P.head   = strtrim(local_git(repo,'rev-parse HEAD'));
P.status = strtrim(local_git(repo,'status --porcelain'));
P.clean  = isempty(P.status);

% ---- currentness + source integrity --------------------------------------
st = olhoffcurrent_currentness('Verbose', false);
P.currentness   = st.state;
P.sourceTree    = st.manifest.treeHash;
P.sourceNFiles  = numel(st.manifest.files);
P.sourceOk      = st.manifest.ok;
P.sourceMismatch= numel(st.manifest.mismatches);
P.sourceMissing = numel(st.manifest.missing);
P.sourceExtra   = numel(st.manifest.extra);

% ---- dispatch / path resolution ------------------------------------------
[guard, gate] = olhoffcurrent_paths(); %#ok<ASGLU>
P.dispatchOk        = gate.ok;
P.dispatchNSymbols  = numel(gate.pathEntries);
P.dispatchBlockers  = numel(gate.blockers);
P.dispatchWarnings  = numel(gate.warnings);
P.resolvedImpl      = gate.resolved;
P.mmasubPath        = which('mmasub');
P.mmaPublished      = ~isempty(strfind(P.mmasubPath, 'mma_published')); %#ok<STREMP>

% ---- forbidden Olhoff trees must not be on the path ----------------------
[repoRel, absolute] = olhoffcurrent_forbidden_paths();
pth = strsplit(path, pathsep);
hits = {};
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

% ---- the scientific lock, read from the resolved production config -------
cfg = olhoffcurrent_config(160, 20);
g = @(p) olh.config.getPath(cfg, p);
P.lock = struct( ...
    'p',                 g('material.stiffness.p'), ...
    'pContinuation',     g('material.stiffness.continuation.enabled'), ...
    'massModel',         g('material.mass.model'), ...
    'q',                 g('material.mass.q'), ...
    'filterType',        g('filter.type'), ...
    'filterApplyTo',     g('filter.applyTo'), ...
    'radiusPhysical',    g('filter.radiusPhysical'), ...
    'projection',        g('projection.enabled'), ...
    'multMethod',        g('multiplicity.method'), ...
    'subspaceSize',      g('multiplicity.subspaceSize'), ...
    'diagonalOffsets',   g('multiplicity.diagonalOffsets'), ...
    'offDiagonal',       g('multiplicity.offDiagonal'), ...
    'mmaVariant',        g('optimizer.inner.variant'), ...
    'innerVariable',     g('optimizer.inner.variable'), ...
    'eigenSolver',       g('eigen.solver'), ...
    'targetMode',        g('eigen.targetMode'), ...
    'volumeFraction',    g('design.volumeFraction'), ...
    'moveLevels',        g('move.levels'), ...
    'movePolicy',        g('move.policy'), ...
    'contSignal',        g('move.continuation.signal'), ...
    'contWindow',        g('move.continuation.window'), ...
    'contTolerance',     g('move.continuation.tolerance'), ...
    'stopNorm',          g('stop.norm'), ...
    'stopToleranceRule', g('stop.toleranceRule'), ...
    'stopTolerance160',  g('stop.tolerance'));
P.sensitivityFilterWins = strcmp(P.lock.filterType,'sensitivity');

% ---- tolerance law identity: tol(NE) used by the frozen rule -------------
P.tolCheck = struct();
meshes = [160 20; 240 30; 320 40; 400 50];
for i = 1:size(meshes,1)
    c  = olhoffcurrent_config(meshes(i,1), meshes(i,2));
    NE = meshes(i,1)*meshes(i,2);
    P.tolCheck(i).mesh = meshes(i,:);
    P.tolCheck(i).NE   = NE;
    P.tolCheck(i).stopTolerance = olh.config.getPath(c,'stop.tolerance');
    P.tolCheck(i).frozenTol     = 0.05*sqrt(NE/3200);
    P.tolCheck(i).identical     = ...
        olh.config.getPath(c,'stop.tolerance') == 0.05*sqrt(NE/3200);
end
P.toleranceLawIdentical = all([P.tolCheck.identical]);

% ---- prior evidence on disk ---------------------------------------------
P.evidence = struct('path',{},'exists',{},'bytes',{},'sha256',{});
cand = { ...
  'analysis/OlhoffCurrent/evidence/move_activity_400/P400_400x50_trajectory.mat'
  'analysis/OlhoffCurrent/evidence/move_activity_400/F400_400x50_trajectory.mat'
  'analysis/OlhoffCurrent/diagnostics/move_stop/runs/baseline_160x20_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/move_stop/runs/baseline_320x40_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/move_stop/runs/fixedmove_160x20_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/move_stop/runs/fixedmove_320x40_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/move_transition/runs/armP_160x20_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/move_transition/runs/armP_320x40_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/move_transition/runs/armU_160x20_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/move_transition/runs/armU_320x40_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/move_activity_400/runs/P400_400x50_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/move_activity_400/runs/F400_400x50_iterations.csv'
  'analysis/OlhoffCurrent/diagnostics/two_branch_maturity_240/runs/runD_240x30.mat'
  'analysis/OlhoffCurrent/diagnostics/dynamical_regime/runs/runB_320x40.mat'
  'analysis/OlhoffCurrent/diagnostics/fixedmove_400_dynamics/runs/runC_400x50.mat'
  'analysis/OlhoffCurrent/diagnostics/fixedmove_400_dynamics/evidence/fm_analysis.mat'
  'analysis/OlhoffCurrent/diagnostics/two_branch_maturity_240/evidence/tb_analysis.mat'};
for i = 1:numel(cand)
    f = fullfile(repo, cand{i});
    P.evidence(i).path   = cand{i};
    P.evidence(i).exists = isfile(f);
    if P.evidence(i).exists
        d = dir(f); P.evidence(i).bytes = d.bytes;
        P.evidence(i).sha256 = olhoffcurrent_sha256_file(f);
    else
        P.evidence(i).bytes = NaN; P.evidence(i).sha256 = '';
    end
end

% ---- evidence gate on the one study that declares evidence ---------------
try
    eg = olhoffcurrent_evidence_gate( ...
        fullfile(root,'diagnostics','move_activity_400'), 'Verbose', false);
    P.moveActivity400Gate = sprintf('ok=%d required=%d present=%d missing=%d mismatch=%d', eg.ok, eg.nRequired, eg.nPresent, eg.nMissing, eg.nMismatch);
catch ME
    P.moveActivity400Gate = ['GATE_ERROR: ' ME.message];
end

% ---- frozen preregistration hashes ---------------------------------------
P.frozen = struct('path',{},'sha256',{});
fz = {'diagnostics/two_branch_maturity_240/PREREGISTRATION.md'
      'diagnostics/two_branch_maturity_240/evidence/PREREGISTRATION.frozen'
      'diagnostics/two_branch_maturity_240/scripts/tb_branches.m'
      'diagnostics/dynamical_regime/PREREGISTRATION.md'
      'diagnostics/dynamical_regime/scripts/dr_telemetry.m'
      'diagnostics/dynamical_regime/scripts/dr_dyn.m'
      'diagnostics/fixedmove_400_dynamics/PREREGISTRATION.md'};
for i = 1:numel(fz)
    f = fullfile(root, fz{i});
    P.frozen(i).path = fz{i};
    if isfile(f), P.frozen(i).sha256 = olhoffcurrent_sha256_file(f);
    else,         P.frozen(i).sha256 = 'MISSING'; end
end

% ---- overall -------------------------------------------------------------
P.gatePass = strcmp(P.currentness,'CURRENT') && P.sourceOk && P.dispatchOk && ...
             P.mmaPublished && P.sensitivityFilterWins && P.forbiddenAbsent && ...
             P.toleranceLawIdentical;

outF = fullfile(study,'evidence',sprintf('provenance_%s.json',tag));
fid = fopen(outF,'w'); fwrite(fid, jsonencode(P,'PrettyPrint',true)); fclose(fid);

fprintf('[cv_provenance:%s]\n', tag);
fprintf('  branch=%s head=%s clean=%d\n', P.branch, P.head, P.clean);
fprintf('  matlab=%s\n', P.matlab);
fprintf('  currentness=%s sourceOk=%d tree=%s (%d files)\n', ...
    P.currentness, P.sourceOk, P.sourceTree, P.sourceNFiles);
fprintf('  dispatch ok=%d symbols=%d blockers=%d warnings=%d\n', ...
    P.dispatchOk, P.dispatchNSymbols, P.dispatchBlockers, P.dispatchWarnings);
fprintf('  mmaPublished=%d sensFilter=%d forbiddenAbsent=%d tolLaw=%d\n', ...
    P.mmaPublished, P.sensitivityFilterWins, P.forbiddenAbsent, P.toleranceLawIdentical);
fprintf('  move.levels=%s policy=%s signal=%s W=%d tol=%g\n', ...
    mat2str(P.lock.moveLevels), P.lock.movePolicy, P.lock.contSignal, ...
    P.lock.contWindow, P.lock.contTolerance);
fprintf('  evidence present: %d / %d\n', sum([P.evidence.exists]), numel(P.evidence));
for i = 1:numel(P.evidence)
    if ~P.evidence(i).exists
        fprintf('    MISSING  %s\n', P.evidence(i).path);
    end
end
fprintf('  move_activity_400 evidence gate: %s\n', P.moveActivity400Gate);
fprintf('  GATE PASS = %d\n', P.gatePass);
fprintf('  wrote %s\n', outF);
end

function s = local_git(repo, args)
[st, s] = system(sprintf('cd %s && git %s', repo, args));
if st ~= 0, s = sprintf('<git failed: %s>', strtrim(s)); end
end
