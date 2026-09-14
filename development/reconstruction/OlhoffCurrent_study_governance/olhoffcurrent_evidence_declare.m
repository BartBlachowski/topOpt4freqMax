function D = olhoffcurrent_evidence_declare(studyDir, study, items, varargin)
%OLHOFFCURRENT_EVIDENCE_DECLARE  Write a study's EVIDENCE.json by hashing reality.
%
%   D = OLHOFFCURRENT_EVIDENCE_DECLARE(studyDir, study, items) records the raw
%   scientific evidence a diagnostic depends on, so OLHOFFCURRENT_EVIDENCE_GATE
%   can later assert it is still there and still unaltered.
%
%   items is an N x 2 or N x 3 cell array:
%
%       { relativePath, class }
%       { relativePath, class, description }
%
%   class is 'required' | 'optional' | 'scratch'.
%
%   Sizes, SHA-256 digests, dimensions and precision are MEASURED FROM THE FILE,
%   never taken on trust from the caller.  A declaration that a required file
%   exists is therefore not writable unless it actually does -- you cannot
%   freeze a study whose evidence was never produced.
%
%   For MAT-files the variable dimensions and classes are read with `whos -file`
%   and recorded, so a later reader knows the trajectory's shape and precision
%   without opening a possibly-huge file.
%
%   Options:
%     'EvidenceRoot' repo-relative durable location for large raw evidence
%                    (default 'analysis/OlhoffCurrent/evidence/<study>')
%     'RepoRoot'     (default derived)
%     'Extra'        struct of additional metadata to embed
%
%   See also OLHOFFCURRENT_EVIDENCE_GATE.

p = inputParser();
p.addParameter('EvidenceRoot', '', @(v) ischar(v) || isstring(v));
p.addParameter('RepoRoot', '', @(v) ischar(v) || isstring(v));
p.addParameter('Extra', struct(), @isstruct);
p.parse(varargin{:});

root = olhoffcurrent_root();
repo = char(p.Results.RepoRoot);
if isempty(repo); repo = fileparts(fileparts(root)); end
evRoot = char(p.Results.EvidenceRoot);
if isempty(evRoot)
    evRoot = fullfile('analysis','OlhoffCurrent','evidence', char(study));
end
evRoot = strrep(evRoot, filesep, '/');

arts = {};
for i = 1:size(items,1)
    rel = char(items{i,1});
    cls = lower(char(items{i,2}));
    desc = '';
    if size(items,2) >= 3 && ~isempty(items{i,3}); desc = char(items{i,3}); end

    cand = {fullfile(repo, evRoot, rel), fullfile(studyDir, rel), fullfile(repo, rel)};
    fp = '';
    for c = 1:numel(cand)
        if exist(cand{c}, 'file') == 2; fp = cand{c}; break; end
    end

    a = struct('path', rel, 'class', cls, 'description', desc);
    if isempty(fp)
        % Refuse to declare a REQUIRED artifact that is not on disk.  Writing
        % such a record would create exactly the fiction this mechanism exists
        % to prevent: a manifest asserting evidence that was never produced.
        if strcmp(cls, 'required')
            error('olhoffcurrent_evidence_declare:RequiredMissing', ...
                ['cannot declare REQUIRED artifact "%s": not found under\n' ...
                 '  %s\n  %s\n' ...
                 'Produce the evidence before freezing the study.'], ...
                rel, cand{1}, cand{2});
        end
        a.bytes = 0; a.sha256 = ''; a.present = false;
    else
        d = dir(fp);
        a.bytes   = d(1).bytes;
        a.sha256  = olhoffcurrent_sha256_file(fp);
        a.present = true;
        [~,~,ext] = fileparts(fp);
        a.format  = local_format(ext);
        if strcmpi(ext, '.mat')
            W = whos('-file', fp);
            v = struct('name', {}, 'size', {}, 'class', {}, 'bytes', {});
            for k = 1:numel(W)
                v(end+1) = struct('name', W(k).name, 'size', W(k).size, ...
                                  'class', W(k).class, 'bytes', W(k).bytes); %#ok<AGROW>
            end
            a.variables = v;
        end
    end
    arts{end+1} = a; %#ok<AGROW>
end

D = struct();
D.schema       = 'olhoff_current_evidence/1';
D.study        = char(study);
D.generated    = char(datetime('now','TimeZone','local','Format','yyyy-MM-dd''T''HH:mm:ssXXX'));
D.evidenceRoot = evRoot;
D.implementation = 'analysis/OlhoffCurrent';
D.sourceTree   = local_treeHash();
D.matlab       = version;
fn = fieldnames(p.Results.Extra);
for i = 1:numel(fn); D.(fn{i}) = p.Results.Extra.(fn{i}); end
D.artifacts    = arts;

if exist(studyDir, 'dir') ~= 7; mkdir(studyDir); end
fid = fopen(fullfile(studyDir, 'EVIDENCE.json'), 'w');
c = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(D, 'PrettyPrint', true));
fprintf('[evidence_declare] %s  (%d artifacts)\n', ...
        fullfile(studyDir,'EVIDENCE.json'), numel(arts));
end

function f = local_format(ext)
switch lower(ext)
    case '.mat',  f = 'MAT-file (v7.3 = HDF5) -- read with load()';
    case '.csv',  f = 'CSV, comma separated, header row';
    case '.json', f = 'JSON';
    otherwise,    f = ext;
end
end

function h = local_treeHash()
try
    m = olhoffcurrent_source_manifest('Verify', false);
    h = m.treeHash;
catch
    h = '';
end
end
