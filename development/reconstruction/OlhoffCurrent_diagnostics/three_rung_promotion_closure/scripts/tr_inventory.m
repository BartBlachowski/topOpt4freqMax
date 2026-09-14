function out = tr_inventory()
%TR_INVENTORY  Phase 0 inventory + Phase 1 reuse verification.
%
%   Records the host, source and repository state BEFORE anything is changed,
%   and verifies that the completed validation retry is intact so its frozen
%   scientific verdict may be REUSED rather than re-derived.
%
%   SCIENTIFICALLY INERT.  This function hashes files and resolves
%   configurations.  It runs no optimization.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root); addpath(here);
guard = olhoffcurrent_paths(); %#ok<NASGU>

out = struct();
out.generated = char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssXXX','TimeZone','local'));
out.matlab = version;
out.threadsDefault = maxNumCompThreads;
out.computer = computer;

% ---- Phase 0: source + production state --------------------------------
man = olhoffcurrent_source_manifest('Verify', true);
out.implTree = man.treeHash;
out.implNFiles = man.nFiles;
out.implOk = man.ok;
st = olhoffcurrent_currentness('Verbose', false);
out.currentness = st.state;

prodCfg = olhoffcurrent_config(320, 40);
out.prodCfgHash320 = olhoffcurrent_config_hash(prodCfg);
out.prodPolicy = struct( ...
    'levels',  olh.config.getPath(prodCfg,'move.levels'), ...
    'signal',  olh.config.getPath(prodCfg,'move.continuation.signal'), ...
    'stopRule',olh.config.getPath(prodCfg,'stop.rule'));
info = olhoffcurrent_preset();
out.presetName = info.name;
out.upstreamPreset = info.upstreamPreset;

% ---- the A/B implementation hashes -------------------------------------
ab = {'+impl/architecture/+olh/+move/exhaustion.m'
      '+impl/architecture/+olh/+move/limit.m'
      '+impl/architecture/olhoffSolve.m'
      '+impl/architecture/+olh/+presets/duOlhoffFrozenM4.m'};
out.controllerHashes = struct('path',{},'sha256',{});
for k = 1:numel(ab)
    out.controllerHashes(end+1) = struct('path', ab{k}, ...
        'sha256', olhoffcurrent_sha256_file(fullfile(root, ab{k}))); %#ok<AGROW>
end

% ---- Phase 1: the validation retry -------------------------------------
r1 = fullfile(root,'diagnostics','three_rung_promotion_validation_retry1');
out.retry1 = struct();
out.retry1.dir = strrep(r1,[repo filesep],'');
g1 = olhoffcurrent_finalization_gate(r1, 'Verbose', false);
out.retry1.gate = struct('ok',g1.ok,'G1',g1.gates.G1,'G2',g1.gates.G2, ...
    'G3',g1.gates.G3,'G4',g1.gates.G4,'G5',g1.gates.G5, ...
    'missing',{g1.missing},'mismatched',{g1.mismatched});
M = jsondecode(fileread(fullfile(r1,'METRICS.json')));
out.retry1.verdicts = M.verdicts;
out.retry1.scientificRuns = M.scientific_runs;
out.retry1.run = M.run;
out.retry1.anchors = M.anchors;
out.retry1.events = M.events;
out.retry1.cost = M.cost;

% the prior stopped study, which must also remain untouched
r0 = fullfile(root,'diagnostics','three_rung_promotion_validation');
out.retry0 = struct('dir', strrep(r0,[repo filesep],''), 'files', struct('name',{},'sha256',{}));
d = dir(fullfile(r0,'*'));
for k = 1:numel(d)
    if d(k).isdir || strcmp(d(k).name,'FINAL_SHA256.txt'), continue; end
    out.retry0.files(end+1) = struct('name', d(k).name, ...
        'sha256', olhoffcurrent_sha256_file(fullfile(r0,d(k).name))); %#ok<AGROW>
end

% ---- Phase 2 precondition: is the C240 artifact here? ------------------
c240 = fullfile(root,'evidence','three_rung_resolution_240','C240x30_trajectory.mat');
out.c240 = struct('path', strrep(c240,[repo filesep],''), ...
    'parentDirExists', isfolder(fileparts(c240)), ...
    'present', exist(c240,'file') == 2, ...
    'expectedSha256','183d7ce60d512fc2c045c3cb575404b00c8f0e223adaf97db86c9eef1fd50b0d', ...
    'expectedBytes', 131203128, 'actualSha256','', 'actualBytes', NaN);
if out.c240.present
    dd = dir(c240);
    out.c240.actualBytes = dd.bytes;
    out.c240.actualSha256 = olhoffcurrent_sha256_file(c240);
end
out.c240.match = out.c240.present && strcmp(out.c240.actualSha256, out.c240.expectedSha256);

out.reuseVerdict = 'VALIDATED_C320_EVIDENCE_REUSE_FAIL';
if g1.ok && strcmp(M.verdicts.policy,'THREE_RUNG_PRODUCTION_POLICY_VALIDATED')
    out.reuseVerdict = 'VALIDATED_C320_EVIDENCE_REUSE_PASS';
end
out.c240Verdict = 'ORIGINAL_C240_EVIDENCE_TRANSFER_FAIL';
if out.c240.match, out.c240Verdict = 'ORIGINAL_C240_EVIDENCE_TRANSFER_PASS'; end

fid = fopen(fullfile(study,'evidence','inventory.json'),'w');
c = onCleanup(@() fclose(fid)); fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));

fprintf('\n== PHASE 0 inventory ==\n');
fprintf('matlab      : %s   threads(default)=%d   %s\n', out.matlab, out.threadsDefault, out.computer);
fprintf('implTree    : %s (ok=%d, n=%d)\n', out.implTree, out.implOk, out.implNFiles);
fprintf('currentness : %s\n', out.currentness);
fprintf('preset      : %s  -> %s\n', out.presetName, out.upstreamPreset);
fprintf('production  : levels=%s signal=%s stop.rule=%s\n', ...
    mat2str(out.prodPolicy.levels), out.prodPolicy.signal, out.prodPolicy.stopRule);
fprintf('prod cfgHash(320x40) : %s\n', out.prodCfgHash320);
for k=1:numel(out.controllerHashes)
    fprintf('  %-52s %s\n', out.controllerHashes(k).path, out.controllerHashes(k).sha256(1:24));
end
fprintf('\n== PHASE 1 retry1 reuse ==\n');
fprintf('gate ok=%d  G1=%d G2=%d G3=%d G4=%d G5=%d\n', out.retry1.gate.ok, ...
    out.retry1.gate.G1,out.retry1.gate.G2,out.retry1.gate.G3,out.retry1.gate.G4,out.retry1.gate.G5);
fprintf('policy   : %s\n', out.retry1.verdicts.policy);
fprintf('prefix   : %s\n', out.retry1.verdicts.prefix);
fprintf('terminat : %s\n', out.retry1.verdicts.termination);
fprintf('run      : %s @%d  inner=%d  cfgHash=%s\n', out.retry1.run.status, ...
    out.retry1.run.nOuter, out.retry1.run.innerTotal, out.retry1.run.cfgHash);
fprintf('VERDICT  : %s\n', out.reuseVerdict);

fprintf('\n== PHASE 2 precondition ==\n');
fprintf('C240 path        : %s\n', out.c240.path);
fprintf('parent dir exists: %d\n', out.c240.parentDirExists);
fprintf('present          : %d\n', out.c240.present);
fprintf('expected sha256  : %s  (%d bytes)\n', out.c240.expectedSha256, out.c240.expectedBytes);
if out.c240.present
    fprintf('actual   sha256  : %s  (%d bytes)\n', out.c240.actualSha256, out.c240.actualBytes);
end
fprintf('VERDICT  : %s\n', out.c240Verdict);
end
