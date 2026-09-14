function [T, meta] = gate_provenance_probes(repo, varargin)
%GATE_PROVENANCE_PROBES  Adversarial probes of the finalization gate's historical-
%   source and current-source rules (diagnostics/provenance_gate_hardening,
%   PREREGISTRATION.md section 4 and PREREGISTRATION_ADDENDUM_1.md; P30 is a disclosed addition).
%
%   [T, meta] = GATE_PROVENANCE_PROBES(repo) builds throwaway COMMITTED histories
%   in a `git clone --shared` of repo's HEAD (a temporary directory) and
%   runs the olhoffcurrent_finalization_gate that is on the MATLAB path against
%   them.  Nothing in repo is written: the clone has its own refs and index.
%
%   T(i): probe, desc, expected, actual, historical, current, reason, pass.
%   Options: 'Only' (cellstr of probe ids), 'KeepClone' (false).
%
%   Scientifically inert: no optimizer, configuration or trajectory is touched;
%   the "promotions" are comment lines appended to source copies in the clone.

p = inputParser();
p.addParameter('Only', {}, @iscell);
p.addParameter('KeepClone', false, @islogical);
p.parse(varargin{:});
only = p.Results.Only;

% MATLAB's system() runs on a TTY: without this, any git command that pages
% (git log) blocks forever in less.  Every call also passes --no-pager.
pagerEntry = getenv('GIT_PAGER');
setenv('GIT_PAGER', 'cat');
pagerRestore = onCleanup(@() local_restoreEnv('GIT_PAGER', pagerEntry)); %#ok<NASGU>

X = struct();
X.repo = char(repo);
[~, head] = system(sprintf('git --no-pager -C "%s" rev-parse HEAD', X.repo)); head = strtrim(head);
[~, before] = system(sprintf('git --no-pager -C "%s" status --porcelain --untracked-files=no', X.repo));
X.clone = [tempname() '_gateprobe'];
if ~p.Results.KeepClone
    cleanup = onCleanup(@() local_rmrf(X.clone)); %#ok<NASGU>
end
% A FULL (non-sparse) checkout: in a sparse checkout git re-derives skip-worktree
% bits from the sparsity patterns, which would defeat probe P28.
local_sh('.', sprintf('git --no-pager clone -q --shared --no-checkout "%s" "%s"', X.repo, X.clone));
local_sh(X.clone, sprintf('git --no-pager -c advice.detachedHead=false checkout -q --detach %s', head));
X.base = head;
X.oc = 'analysis/OlhoffCurrent';
X.srcA = [X.oc '/+impl/architecture/olhoffSolve.m'];
X.srcB = [X.oc '/+impl/architecture/+olh/+move/limit.m'];
X.readme = [X.oc '/README.md'];
X.probe = [X.oc '/+impl/architecture/probe_gate_empty.m'];
X.studyRel = [X.oc '/diagnostics/_gate_probe'];
X.study = fullfile(X.clone, X.studyRel);
X.evRel = [X.oc '/evidence/_gate_probe/data'];
X.EMPTY = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
X.fab = repmat('c', 1, 64);

P = local_probes();
T = struct('probe', {}, 'desc', {}, 'expected', {}, 'actual', {}, 'historical', {}, ...
           'current', {}, 'reason', {}, 'pass', {});
for i = 1:size(P, 1)
    id = P{i, 1};
    if ~isempty(only) && ~any(strcmp(only, id)); continue; end
    local_reset(X);
    try
        st = feval(P{i, 4}, X);
        r = local_reason(st);
        T(end+1) = struct('probe', id, 'desc', P{i, 2}, 'expected', P{i, 3}, 'actual', st.ok, ...
            'historical', local_status(st, 'historicalSource'), 'current', local_status(st, 'currentSource'), ...
            'reason', r, 'pass', isequal(st.ok, P{i, 3})); %#ok<AGROW>
        fprintf('  probe %-4s expected=%d actual=%d\n', id, P{i, 3}, st.ok);
    catch ME
        fprintf('  probe %-4s HARNESS ERROR: %s\n', id, ME.message);
        T(end+1) = struct('probe', id, 'desc', P{i, 2}, 'expected', P{i, 3}, 'actual', NaN, ...
            'historical', '', 'current', '', 'reason', ['HARNESS ERROR: ' ME.message], 'pass', false); %#ok<AGROW>
    end
end
[~, after] = system(sprintf('git --no-pager -C "%s" status --porcelain --untracked-files=no', X.repo));
[~, headAfter] = system(sprintf('git --no-pager -C "%s" rev-parse HEAD', X.repo));
meta = struct('repo', X.repo, 'head', head, 'clone', X.clone, ...
    'repoTrackedStatusUnchanged', strcmp(before, after), 'repoHeadUnchanged', strcmp(head, strtrim(headAfter)), ...
    'nProbes', numel(T), 'nAsExpected', sum([T.pass]), 'gate', which('olhoffcurrent_finalization_gate'));
end

% =============================================================================
function P = local_probes()
% id, description, expected ok, implementation
P = {
 'P0',  'compliant study, no source lines', true, @p0
 'P1',  'genuine olhoffSolve.m digest at the freeze commit, after a promotion', true, @p1
 'P2',  'single fabricated digest on a source path', false, @p2
 'P3',  'limit.m freeze digest placed on the olhoffSolve.m line (wrong path, valid digest)', false, @p3
 'P4',  'non-source README.md carrying its freeze digest after it changed', false, @p4
 'P5',  'fabricated digest FOLLOWED by the genuine digest, same path', false, @p5
 'P6',  'genuine digest followed by a fabricated digest, same path', false, @p6
 'P7',  'empty-file digest on olhoffSolve.m (non-empty at the freeze commit)', false, @p7
 'P8',  'genuine superseded line; +impl locally edited, manifest unchanged', false, @p8
 'P9',  'empty digest; probe path deleted before the freeze commit, re-added after', false, @p9
 'P10', 'genuine superseded line; +impl edited AND SOURCE_MANIFEST.json regenerated', false, @p10
 'P11', 'two identical genuine lines, same path', true, @p11
 'P12', 'olhoffSolve.m digest from an OLDER commit, not the freeze commit', false, @p12
 'P13', 'digest of a non-source file placed on a source-path line', false, @p13
 'P14', 'malformed: uppercase 64-hex digest on a source line', false, @p14
 'P15', 'malformed: 63-hex digest on a source line', false, @p15
 'P16', 'non-canonical source path (architecture/../architecture) with genuine digest', false, @p16
 'P17', 'missing: source path absent from working tree and history', false, @p17
 'P18', 'empty digest; path never existed at the freeze commit, added later', false, @p18
 'P19', 'empty digest; genuinely EMPTY committed file at the freeze commit, non-empty now', true, @p19
 'P20', 'empty digest; path non-empty at the freeze commit, changed later', false, @p20
 'P21', 'study-local shadow copy of a source path with matching fabricated digest', false, @p21
 'P22', 'genuine superseded line; EVIDENCE.json declares no source tree', false, @p22
 'P23', 'genuine superseded line; declared source tree is not the freeze-commit tree', false, @p23
 'P24', 'genuine superseded lines; hash file carries an uncommitted extra line', false, @p24
 'P25', 'SOURCE_MANIFEST.json tampered (consistent-looking rows), +impl clean', false, @p25
 'P26', 'untracked extra source file in +impl, manifest regenerated', false, @p26
 'P27', '+impl file deleted in the working tree', false, @p27
 'P28', '+impl edit + regenerated manifest hidden with git update-index --skip-worktree', false, @p28
 'P29', 'committed +impl change + committed manifest, PROVENANCE.md tree not updated', false, @p29
 'P30', 'case-variant source path resolving to a study-local shadow copy (addition)', false, @p30
 'P31', 'OlhoffCurrent//+impl spelling + study-local shadow copy (ADDENDUM_1)', false, @p31
 'P32', 'OlhoffCurrent/./+impl spelling + study-local shadow copy (ADDENDUM_1)', false, @p32
 'P33', 'study-relative ../../+impl path to the REAL production file, current digest (ADDENDUM_1)', false, @p33
 'P34', '+impl//architecture spelling with the genuine freeze digest (ADDENDUM_1)', false, @p34
 'P35', 'freeze-commit blob forged with git replace, fabricated digest (ADDENDUM_1)', false, @p35
 'P36', 'dirty +impl, GIT_DIR redirected to a clone committing the edit (ADDENDUM_1)', false, @p36
 'P37', 'dirty +impl, HEAD blobs substituted with git replace (ADDENDUM_1)', false, @p37
 'P38', 'study-local symlink to the real production file, current digest (ADDENDUM_1)', false, @p38
};
end

% ---- probe bodies -------------------------------------------------------------
function st = p0(X)
local_study(X, {}); local_commit(X, 'study'); st = local_gate(X);
end
function st = p1(X)
st = local_promotionScenario(X, @(d) {hl(d.A, X.srcA)});
end
function st = p2(X)
st = local_promotionScenario(X, @(d) {hl(X.fab, X.srcA)});
end
function st = p3(X)
st = local_promotionScenario(X, @(d) {hl(d.B, X.srcA)});
end
function st = p4(X)
st = local_promotionScenario(X, @(d) {hl(d.R, X.readme)});
end
function st = p5(X)
st = local_promotionScenario(X, @(d) {hl(X.fab, X.srcA), hl(d.A, X.srcA)});
end
function st = p6(X)
st = local_promotionScenario(X, @(d) {hl(d.A, X.srcA), hl(X.fab, X.srcA)});
end
function st = p7(X)
st = local_promotionScenario(X, @(d) {hl(X.EMPTY, X.srcA)});
end
function st = p8(X)
st = local_promotionScenario(X, @(d) {hl(d.A, X.srcA)}, 'post', @(X) local_append(X, X.srcA, '% local edit'));
end
function st = p9(X)
local_writeFile(X, X.probe, 'y'); local_promote(X, 'add probe');
delete(fullfile(X.clone, X.probe)); local_promote(X, 'delete probe');
local_study(X, {hl(X.EMPTY, X.probe)}); local_commit(X, 'freeze');
local_writeFile(X, X.probe, 'y'); local_promote(X, 're-add probe');
st = local_gate(X);
end
function st = p10(X)
st = local_promotionScenario(X, @(d) {hl(d.A, X.srcA)}, 'post', @(X) local_editAndRegen(X, X.srcA));
end
function st = p11(X)
st = local_promotionScenario(X, @(d) {hl(d.A, X.srcA), hl(d.A, X.srcA)});
end
function st = p12(X)
[~, out] = system(sprintf('git --no-pager -C "%s" log --format=%%H %s -- %s', X.clone, X.base, X.srcA));
C = strsplit(strtrim(out));
dNow = local_blobDigest(X, X.base, X.srcA);
old = '';
for k = numel(C):-1:1
    h = local_blobDigest(X, C{k}, X.srcA);
    if ~isempty(h) && ~strcmp(h, dNow); old = h; break; end
end
assert(~isempty(old), 'probe:P12', 'no older distinct version of olhoffSolve.m in history');
st = local_promotionScenario(X, @(d) {hl(old, X.srcA)});
end
function st = p13(X)
st = local_promotionScenario(X, @(d) {hl(d.R, X.srcA)});
end
function st = p14(X)
st = local_promotionScenario(X, @(d) {hl(upper(d.A), X.srcA)});
end
function st = p15(X)
st = local_promotionScenario(X, @(d) {hl(d.A(1:63), X.srcA)});
end
function st = p16(X)
st = local_promotionScenario(X, @(d) {hl(d.A, strrep(X.srcA, 'architecture/', 'architecture/../architecture/'))});
end
function st = p17(X)
st = local_promotionScenario(X, @(d) {hl(X.fab, [X.oc '/+impl/architecture/never_existed_probe.m'])});
end
function st = p18(X)
local_study(X, {hl(X.EMPTY, X.probe)}); local_commit(X, 'freeze');
local_writeFile(X, X.probe, 'y'); local_promote(X, 'add probe');
st = local_gate(X);
end
function st = p19(X)
local_writeFile(X, X.probe, ''); local_promote(X, 'add EMPTY probe');
local_study(X, {hl(X.EMPTY, X.probe)}); local_commit(X, 'freeze');
local_writeFile(X, X.probe, 'y'); local_promote(X, 'fill probe');
st = local_gate(X);
end
function st = p20(X)
local_writeFile(X, X.probe, 'z'); local_promote(X, 'add probe');
local_study(X, {hl(X.EMPTY, X.probe)}); local_commit(X, 'freeze');
local_writeFile(X, X.probe, 'y'); local_promote(X, 'change probe');
st = local_gate(X);
end
function st = p21(X)
fake = sprintf('%% fabricated shadow of olhoffSolve.m\n');
shadowRel = [X.studyRel '/' X.srcA];
local_writeFile(X, shadowRel, fake);
local_study(X, {hl(local_sha(uint8(fake)), X.srcA)}); local_commit(X, 'freeze');
st = local_gate(X);
end
function st = p22(X)
st = local_promotionScenario(X, @(d) {hl(d.A, X.srcA)}, 'tree', 'remove');
end
function st = p23(X)
st = local_promotionScenario(X, @(d) {hl(d.A, X.srcA)}, 'tree', repmat('d', 1, 64));
end
function st = p24(X)
st = local_promotionScenario(X, @(d) {hl(d.A, X.srcA)}, 'post', ...
    @(X) local_append(X, [X.studyRel '/FINAL_SHA256.txt'], ...
        hl(local_fileSha(fullfile(X.study, 'EVIDENCE.json')), 'EVIDENCE.json')));
end
function st = p25(X)
local_study(X, {}); local_commit(X, 'study');
mf = fullfile(X.clone, X.oc, 'SOURCE_MANIFEST.json');
M = jsondecode(fileread(mf));
M.files(1).sha256 = repmat('e', 1, 64);
rows = arrayfun(@(f) sprintf('%s  %s', f.path, f.sha256), M.files, 'UniformOutput', false);
M.tree_sha256 = local_sha(uint8(strjoin(sort(rows(:)), newline)));
fid = fopen(mf, 'w'); fwrite(fid, jsonencode(M, 'PrettyPrint', true)); fclose(fid);
st = local_gate(X);
end
function st = p26(X)
local_study(X, {}); local_commit(X, 'study');
local_writeFile(X, [X.oc '/+impl/architecture/extra_probe.m'], sprintf('function extra_probe()\nend\n'));
local_writeManifest(X);
st = local_gate(X);
end
function st = p27(X)
local_study(X, {}); local_commit(X, 'study');
delete(fullfile(X.clone, X.srcB));
st = local_gate(X);
end
function st = p28(X)
local_study(X, {}); local_commit(X, 'study');
local_editAndRegen(X, X.srcA);
local_sh(X.clone, sprintf('git --no-pager update-index --skip-worktree -- %s %s/SOURCE_MANIFEST.json', X.srcA, X.oc));
[~, s] = system(sprintf('git --no-pager -C "%s" status --porcelain -- %s/+impl %s/SOURCE_MANIFEST.json', X.clone, X.oc, X.oc));
assert(isempty(strtrim(s)), 'probe:P28', 'skip-worktree did not hide the edit from git status');
st = local_gate(X);
end
function st = p29(X)
local_study(X, {}); local_commit(X, 'study');
local_append(X, X.srcA, '% committed change');
local_writeManifest(X);
local_sh(X.clone, sprintf('git --no-pager add -- %s %s/SOURCE_MANIFEST.json', X.srcA, X.oc));
local_sh(X.clone, local_commitCmd('source change without provenance'));
st = local_gate(X);
end
function st = p30(X)
fake = sprintf('%% fabricated shadow of olhoffSolve.m\n');
variant = strrep(strrep(X.srcA, 'OlhoffCurrent', 'olhoffcurrent'), '+impl', '+IMPL');
local_writeFile(X, [X.studyRel '/' variant], fake);
local_study(X, {hl(local_sha(uint8(fake)), variant)}); local_commit(X, 'freeze');
st = local_gate(X);
end

function st = p31(X)
st = local_shadowSpelling(X, strrep(X.srcA, 'OlhoffCurrent/+impl', 'OlhoffCurrent//+impl'));
end
function st = p32(X)
st = local_shadowSpelling(X, strrep(X.srcA, 'OlhoffCurrent/+impl', 'OlhoffCurrent/./+impl'));
end
function st = p33(X)
rel = '../../+impl/architecture/olhoffSolve.m';          % relative to analysis/OlhoffCurrent/diagnostics/<study>
local_study(X, {hl(local_fileSha(fullfile(X.clone, X.srcA)), rel)}); local_commit(X, 'freeze');
st = local_gate(X);
end
function st = p34(X)
st = local_promotionScenario(X, @(d) {hl(d.A, strrep(X.srcA, '+impl/architecture', '+impl//architecture'))});
end
function st = p35(X)
fake = sprintf('%% forged historical content\n');
st = local_promotionScenario(X, @(d) {hl(local_sha(uint8(fake)), X.srcA)}, 'post', @(X) local_replaceAtFreeze(X, fake));
end
function st = p36(X)
local_study(X, {}); local_commit(X, 'study');
local_dirtyWithProvenance(X);
B = [X.clone '_B'];
c = onCleanup(@() local_rmrf(B)); %#ok<NASGU>
local_sh('.', sprintf('git --no-pager clone -q --shared "%s" "%s"', X.clone, B));
for rel = {X.srcA, [X.oc '/SOURCE_MANIFEST.json'], [X.oc '/PROVENANCE.md']}
    copyfile(fullfile(X.clone, rel{1}), fullfile(B, rel{1}));
end
local_sh(B, sprintf('git --no-pager add -- %s %s/SOURCE_MANIFEST.json %s/PROVENANCE.md', X.srcA, X.oc, X.oc));
local_sh(B, local_commitCmd('edit committed elsewhere'));
entry = getenv('GIT_DIR');
setenv('GIT_DIR', fullfile(B, '.git'));
restore = onCleanup(@() local_restoreEnv('GIT_DIR', entry)); %#ok<NASGU>
st = local_gate(X);
end
function st = p37(X)
local_study(X, {}); local_commit(X, 'study');
local_dirtyWithProvenance(X);
for rel = {X.srcA, [X.oc '/SOURCE_MANIFEST.json']}
    [~, old] = system(sprintf('git --no-pager -C "%s" rev-parse "HEAD:%s"', X.clone, rel{1}));
    [~, new] = system(sprintf('git --no-pager -C "%s" hash-object -w -- "%s"', X.clone, rel{1}));
    local_sh(X.clone, sprintf('git --no-pager replace -f %s %s', strtrim(old), strtrim(new)));
end
st = local_gate(X);
end
function st = p38(X)
local_study(X, {});
target = fullfile(X.clone, X.srcA);
link = fullfile(X.study, 'linked_solver.m');
local_sh('.', sprintf('ln -s "%s" "%s"', target, link));
fid = fopen(fullfile(X.study, 'FINAL_SHA256.txt'), 'a');
fprintf(fid, '%s\n', hl(local_fileSha(target), 'linked_solver.m')); fclose(fid);
local_commit(X, 'freeze');
st = local_gate(X);
end

% ---- scenario helpers ---------------------------------------------------------
function st = local_shadowSpelling(X, spelled)
fake = sprintf('%% fabricated shadow of olhoffSolve.m\n');
local_writeFile(X, [X.studyRel '/' X.srcA], fake);
local_study(X, {hl(local_sha(uint8(fake)), spelled)}); local_commit(X, 'freeze');
st = local_gate(X);
end

function local_dirtyWithProvenance(X)
%LOCAL_DIRTYWITHPROVENANCE  Edit +impl, regenerate the manifest and the PROVENANCE row (uncommitted).
local_append(X, X.srcA, '% local edit');
tree = local_writeManifest(X);
pv = fullfile(X.clone, X.oc, 'PROVENANCE.md');
txt = regexprep(fileread(pv), '(\*\*Source tree SHA-256\*\*\s*\|\s*`)[0-9a-f]{64}(`)', ['$1' tree '$2']);
fid = fopen(pv, 'w'); fwrite(fid, txt); fclose(fid);
end

function local_replaceAtFreeze(X, fake)
%LOCAL_REPLACEATFREEZE  Substitute the freeze-commit blob of olhoffSolve.m via git replace.
[~, fz] = system(sprintf('git --no-pager -C "%s" log -1 --format=%%H HEAD -- %s/FINAL_SHA256.txt', X.clone, X.studyRel));
[~, old] = system(sprintf('git --no-pager -C "%s" rev-parse "%s:%s"', X.clone, strtrim(fz), X.srcA));
tmp = [tempname() '.fake']; fid = fopen(tmp, 'w'); fwrite(fid, fake); fclose(fid);
[~, new] = system(sprintf('git --no-pager -C "%s" hash-object -w -- "%s"', X.clone, tmp)); delete(tmp);
local_sh(X.clone, sprintf('git --no-pager replace -f %s %s', strtrim(old), strtrim(new)));
end

function st = local_promotionScenario(X, linesFn, varargin)
%LOCAL_PROMOTIONSCENARIO  study frozen at the base state, THEN a committed promotion
%   (olhoffSolve.m, limit.m, README.md changed; manifest and PROVENANCE tree updated).
o = struct('post', [], 'tree', '');
for k = 1:2:numel(varargin); o.(varargin{k}) = varargin{k+1}; end
d = struct('A', local_fileSha(fullfile(X.clone, X.srcA)), 'B', local_fileSha(fullfile(X.clone, X.srcB)), ...
           'R', local_fileSha(fullfile(X.clone, X.readme)));
local_study(X, linesFn(d), o.tree);
local_commit(X, 'freeze');
local_append(X, X.srcA, '% probe promotion');
local_append(X, X.srcB, '% probe promotion');
local_append(X, X.readme, 'probe promotion');
local_promote(X, 'promotion');
if ~isempty(o.post); o.post(X); end
st = local_gate(X);
end

function local_study(X, extraLines, treeOpt)
if nargin < 3; treeOpt = ''; end
evAbs = fullfile(X.clone, X.evRel);
if exist(evAbs, 'dir') ~= 7; mkdir(evAbs); end
RHO = zeros(4, 2); %#ok<NASGU>
save(fullfile(evAbs, 'arm_probe_trajectory.mat'), 'RHO', '-v7.3');
if exist(X.study, 'dir') ~= 7; mkdir(X.study); end
tree = local_workingTree(X);
if ~isempty(treeOpt) && ~strcmp(treeOpt, 'remove'); tree = treeOpt; end
evalc(['olhoffcurrent_evidence_declare(X.study, ''_gate_probe'', ' ...
       '{''arm_probe_trajectory.mat'', ''required'', ''probe''}, ''EvidenceRoot'', X.evRel, ' ...
       '''RepoRoot'', X.clone, ''Extra'', struct(''sourceTree'', tree));']);
ev = fullfile(X.study, 'EVIDENCE.json');
if strcmp(treeOpt, 'remove')
    txt = fileread(ev);
    txt2 = regexprep(txt, '\s*"sourceTree":\s*"[0-9a-f]*",', '');
    assert(~strcmp(txt, txt2) && ~isfield(jsondecode(txt2), 'sourceTree'), 'probe:tree', 'sourceTree not removed');
    fid = fopen(ev, 'w'); fwrite(fid, txt2); fclose(fid);
end
L = [{hl(local_fileSha(ev), 'EVIDENCE.json')}, extraLines];
fid = fopen(fullfile(X.study, 'FINAL_SHA256.txt'), 'w');
fprintf(fid, '%s\n', L{:}); fclose(fid);
end

function local_commit(X, msg)
local_sh(X.clone, sprintf('git --no-pager add -A -- %s', X.studyRel));
local_sh(X.clone, local_commitCmd(msg));
end

function local_promote(X, msg)
tree = local_writeManifest(X);
pv = fullfile(X.clone, X.oc, 'PROVENANCE.md');
txt = fileread(pv);
txt2 = regexprep(txt, '(\*\*Source tree SHA-256\*\*\s*\|\s*`)[0-9a-f]{64}(`)', ['$1' tree '$2']);
fid = fopen(pv, 'w'); fwrite(fid, txt2); fclose(fid);
local_sh(X.clone, sprintf('git --no-pager add -A -- %s/+impl %s/SOURCE_MANIFEST.json %s/PROVENANCE.md %s', ...
    X.oc, X.oc, X.oc, X.readme));
local_sh(X.clone, local_commitCmd(msg));
end

function local_editAndRegen(X, rel)
local_append(X, rel, '% local edit');
local_writeManifest(X);
end

function tree = local_writeManifest(X)
%LOCAL_WRITEMANIFEST  SOURCE_MANIFEST.json of the clone's working +impl (same schema
%   and tree algorithm as olhoffcurrent_source_manifest), written into the clone.
impl = fullfile(X.clone, X.oc, '+impl');
W = dir(fullfile(impl, '**', '*')); W = W(~[W.isdir]);
rel = {}; sh = {};
for k = 1:numel(W)
    if olhoffcurrent_is_artifact(W(k).name); continue; end
    fp = fullfile(W(k).folder, W(k).name);
    rel{end+1} = strrep(fp(numel(impl)+2:end), filesep, '/'); %#ok<AGROW>
    sh{end+1} = local_fileSha(fp); %#ok<AGROW>
end
[rel, ix] = sort(rel); sh = sh(ix);
rows = cellfun(@(a, b) sprintf('%s  %s', a, b), rel, sh, 'UniformOutput', false);
tree = local_sha(uint8(strjoin(rows, newline)));
S = struct('manifest_schema', 'olhoff_current_source_manifest/1', 'generated', 'probe', ...
    'root', 'analysis/OlhoffCurrent/+impl', 'n_files', numel(rel), 'tree_sha256', tree, ...
    'files', struct('path', rel, 'sha256', sh));
fid = fopen(fullfile(X.clone, X.oc, 'SOURCE_MANIFEST.json'), 'w');
fprintf(fid, '%s\n', jsonencode(S, 'PrettyPrint', true)); fclose(fid);
end

function tree = local_workingTree(X)
impl = fullfile(X.clone, X.oc, '+impl');
W = dir(fullfile(impl, '**', '*')); W = W(~[W.isdir]);
rel = {}; sh = {};
for k = 1:numel(W)
    if olhoffcurrent_is_artifact(W(k).name); continue; end
    fp = fullfile(W(k).folder, W(k).name);
    rel{end+1} = strrep(fp(numel(impl)+2:end), filesep, '/'); %#ok<AGROW>
    sh{end+1} = local_fileSha(fp); %#ok<AGROW>
end
[rel, ix] = sort(rel); sh = sh(ix);
rows = cellfun(@(a, b) sprintf('%s  %s', a, b), rel, sh, 'UniformOutput', false);
tree = local_sha(uint8(strjoin(rows, newline)));
end

function st = local_gate(X)
st = olhoffcurrent_finalization_gate(X.study, 'Verbose', false, 'RepoRoot', X.clone);
end

function local_reset(X)
[~, refs] = system(sprintf('git --no-pager -C "%s" for-each-ref --format="%%(refname)" refs/replace', X.clone));
refs = strsplit(strtrim(refs));
for k = 1:numel(refs)
    if ~isempty(refs{k}); local_sh(X.clone, sprintf('git --no-pager update-ref -d %s', refs{k})); end
end
local_sh(X.clone, sprintf('git --no-pager update-index --no-skip-worktree -- %s %s/SOURCE_MANIFEST.json', X.srcA, X.oc));
local_sh(X.clone, sprintf('git --no-pager -c advice.detachedHead=false checkout -q -f --detach %s', X.base));
local_sh(X.clone, sprintf('git --no-pager clean -fdxq -- %s/+impl %s %s/evidence/_gate_probe', X.oc, X.studyRel, X.oc));
if exist(X.study, 'dir') == 7; rmdir(X.study, 's'); end
end

% ---- primitives -----------------------------------------------------------------
function s = hl(d, p), s = sprintf('%s  %s', d, p); end

function local_append(X, rel, txt)
fid = fopen(fullfile(X.clone, rel), 'a'); fprintf(fid, '\n%s\n', txt); fclose(fid);
end

function local_writeFile(X, rel, txt)
fp = fullfile(X.clone, rel);
if exist(fileparts(fp), 'dir') ~= 7; mkdir(fileparts(fp)); end
fid = fopen(fp, 'w'); fwrite(fid, txt); fclose(fid);
end

function c = local_commitCmd(msg)
c = sprintf(['git --no-pager -c user.name=probe -c user.email=probe@invalid -c commit.gpgsign=false ' ...
             '-c core.hooksPath=/dev/null commit -q --allow-empty -m "%s"'], msg);
end

function h = local_blobDigest(X, commit, rel)
h = '';
tmp = [tempname() '.blob'];
if system(sprintf('git --no-pager -C "%s" cat-file -e "%s:%s"', X.clone, commit, rel)) ~= 0; return; end
if system(sprintf('git --no-pager -C "%s" cat-file blob "%s:%s" > "%s"', X.clone, commit, rel, tmp)) ~= 0; return; end
h = local_fileSha(tmp); delete(tmp);
end

function h = local_fileSha(fp)
fid = fopen(fp, 'r'); b = fread(fid, Inf, '*uint8'); fclose(fid);
h = local_sha(b);
end

function h = local_sha(bytes)
md = java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(bytes); md.update(uint8(bytes(:))); end
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
end

function s = local_status(st, field)
%LOCAL_STATUS  Status string, tolerant of earlier gate versions without the field.
if isfield(st, field) && isfield(st.(field), 'status'); s = st.(field).status; else; s = 'n/a (gate version)'; end
end

function r = local_reason(st)
parts = {};
if isfield(st, 'malformedSourceLines')
    for i = 1:numel(st.malformedSourceLines)
        parts{end+1} = sprintf('MALFORMED line %d: %s', st.malformedSourceLines(i).lineNo, st.malformedSourceLines(i).reason); %#ok<AGROW>
    end
end
srcPaths = {};
if isfield(st, 'sourceLines')
    srcPaths = {st.sourceLines.path};
    for i = 1:numel(st.sourceLines)
        s = st.sourceLines(i);
        if ~strcmp(s.verdict, 'CURRENT_MATCH')
            parts{end+1} = sprintf('line %d %s: %s', s.lineNo, s.verdict, s.reason); %#ok<AGROW>
        end
    end
end
nonSrc = setdiff([st.missing, st.mismatched], srcPaths);
for i = 1:numel(nonSrc); parts{end+1} = ['MISMATCH/MISSING: ' nonSrc{i}]; end %#ok<AGROW>
if isfield(st, 'duplicatePaths') && ~isempty(st.duplicatePaths)
    parts{end+1} = ['DUPLICATE_PATHS: ' strjoin(st.duplicatePaths, ', ')];
end
if isfield(st, 'currentSource') && ~isempty(st.currentSource.reasons)
    cr = st.currentSource.reasons;
    parts{end+1} = ['CURRENT: ' strjoin(cr(1:min(4, end)), '; ')];
end
g = st.gates; gv = [g.G1 g.G2 g.G3 g.G4 g.G5];
if isfield(g, 'G6'); gv(end+1) = g.G6; end
parts{end+1} = ['G=' sprintf('%d', gv)];
r = strjoin(parts, ' | ');
end

function local_sh(dirp, cmd)
if strcmp(dirp, '.'), full = cmd; else, full = sprintf('cd "%s" && %s', dirp, cmd); end
[s, o] = system([full ' 2>&1']);
if s ~= 0; error('probe:sh', 'command failed (%d): %s\n%s', s, cmd, o); end
end

function local_rmrf(p)
if exist(p, 'dir') == 7; rmdir(p, 's'); end
end

function local_restoreEnv(name, value)
%LOCAL_RESTOREENV  Restore an environment variable; an originally absent one is UNSET
%   (setenv(name, '') would leave an empty value, which git rejects for GIT_DIR).
if isempty(value); unsetenv(name); else; setenv(name, value); end
end
