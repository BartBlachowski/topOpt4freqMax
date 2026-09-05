function report = olhoffm4_verify_import(varargin)
%OLHOFFM4_VERIFY_IMPORT  Prove the imported solver is the audited one, unchanged.
%
%   report = OLHOFFM4_VERIFY_IMPORT() proves, using evidence held inside THIS
%   repository, that the Du-Olhoff (M4) reconstruction which is about to run is
%   the audited implementation and nothing else.  Three things are proved:
%
%     1. INTEGRITY.  Every file under +frozen/ still hashes to the
%        sha256_imported recorded in IMPORT_MANIFEST.json.
%
%     2. ATTESTATION.  For every imported file with no declared modification,
%        the manifest records sha256_imported == sha256_source, i.e. the record
%        itself attests byte-identity to the audited source.
%
%     3. RECONSTRUCTION.  For the one declared modification, the verbatim copy
%        of its audited source hashes to sha256_source, and applying the
%        declared diff to that copy reproduces the file under +frozen/ byte for
%        byte.  So "declared" cannot become a loophole: the delta from the
%        audited source is exhibited, not asserted.
%
%   Together these pin the running solver to the audited source without needing
%   the external source repository, which is unversioned, outside this
%   repository and not under its control.
%
%   THE EXTERNAL SOURCE REPOSITORY IS PROVENANCE, NOT A GATE.  Its present
%   state is measured and reported in report.source_repository_state, and is
%   deliberately NOT part of report.pass.  It was made informational on
%   2026-09-04, after that directory was found reverted to a pre-import
%   generation (seven audited files absent, four differing) while every file
%   under +frozen/ still hashed exactly to the manifest.  A rollback in an
%   unversioned scratch directory cannot change what this repository runs, so
%   it must not be able to change whether this repository may run it.  See
%   IMPORT_MANIFEST.json -> source_repository_status.
%
%   Name/value options:
%     'Verbose'  (default true)  print a summary
%
%   report.pass is true only when integrity, attestation and reconstruction all
%   hold AND the dispatch gate resolves every owned function inside this import.
%
%   See also OLHOFFM4_APPLY_UNIFIED_DIFF, OLHOFFM4_PATHS, CONFBENCH_PREFLIGHT.

p = inputParser();
p.addParameter('Verbose', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
verbose = p.Results.Verbose;

root = olhoffm4_root();
man = jsondecode(fileread(fullfile(root, 'IMPORT_MANIFEST.json')));
repoRoot = fileparts(fileparts(root));

report = struct();
report.manifest_schema = man.manifest_schema;
report.import_datetime = man.import_datetime_local;
report.checked = 0;

% Blocking findings.  Each is empty when the corresponding proof holds.
report.imported_hash_mismatches  = {};   % 1. integrity
report.unattested_files          = {};   % 2. attestation
report.declared_patch_mismatches = {};   % 3. reconstruction
report.patch_checks              = {};   % what reconstruction actually proved

report.declared_modifications = {man.modifications.file};
declaredDest = fullfile('analysis', 'OlhoffM4Reconstruction', report.declared_modifications);

files = man.files;

% ---- 1. INTEGRITY, and 2. ATTESTATION -----------------------------------
for i = 1:numel(files)
    e = files(i);
    dst = fullfile(repoRoot, e.destination_path);
    if exist(dst, 'file') ~= 2
        report.imported_hash_mismatches{end+1} = sprintf('%s MISSING', e.destination_path); %#ok<AGROW>
    else
        hNow = olhoffm4_sha256_file(dst);
        report.checked = report.checked + 1;
        if ~strcmp(hNow, e.sha256_imported)
            report.imported_hash_mismatches{end+1} = sprintf('%s: %s != manifest %s', ...
                e.destination_path, hNow, e.sha256_imported); %#ok<AGROW>
        end
    end

    % Every imported file is either byte-identical to the audited source by the
    % record, or carries a declared modification proved below.  Nothing else.
    if ~any(strcmp(e.destination_path, declaredDest)) && ...
            ~strcmp(e.sha256_imported, e.sha256_source)
        report.unattested_files{end+1} = sprintf( ...
            ['%s: the manifest does not attest byte-identity to the audited ' ...
             'source (imported %s != source %s) and no modification is declared'], ...
            e.destination_path, e.sha256_imported, e.sha256_source); %#ok<AGROW>
    end
end

% ---- 3. RECONSTRUCTION of every declared modification -------------------
for i = 1:numel(man.modifications)
    m = man.modifications(i);
    dest = fullfile('analysis', 'OlhoffM4Reconstruction', m.file);
    k = find(strcmp({files.destination_path}, dest), 1);
    if isempty(k)
        report.declared_patch_mismatches{end+1} = sprintf( ...
            '%s: declared as modified but absent from the manifest file list', m.file); %#ok<AGROW>
        continue
    end
    verbatim = fullfile(root, m.source_verbatim_copy);
    diffPath = fullfile(root, m.diff);
    if exist(verbatim, 'file') ~= 2 || exist(diffPath, 'file') ~= 2
        report.declared_patch_mismatches{end+1} = sprintf( ...
            ['%s: the declared modification cannot be checked -- verbatim source ' ...
             'copy or diff is missing (%s, %s)'], m.file, m.source_verbatim_copy, m.diff); %#ok<AGROW>
        continue
    end

    hVerbatim = olhoffm4_sha256_file(verbatim);
    if ~strcmp(hVerbatim, files(k).sha256_source)
        report.declared_patch_mismatches{end+1} = sprintf( ...
            ['%s: the verbatim copy of the audited source hashes %s, but the ' ...
             'manifest records the audited source as %s'], ...
            m.file, hVerbatim, files(k).sha256_source); %#ok<AGROW>
        continue
    end

    try
        rebuilt = olhoffm4_apply_unified_diff( ...
            char(olhoffm4_read_bytes(verbatim).'), char(olhoffm4_read_bytes(diffPath).'));
    catch ME
        report.declared_patch_mismatches{end+1} = sprintf( ...
            '%s: the declared diff does not apply to the verbatim source copy (%s)', ...
            m.file, ME.message); %#ok<AGROW>
        continue
    end

    hRebuilt = olhoffm4_sha256_bytes(rebuilt);
    if ~strcmp(hRebuilt, files(k).sha256_imported)
        report.declared_patch_mismatches{end+1} = sprintf( ...
            ['%s: audited source + declared diff hashes %s, but the file that ' ...
             'runs is %s -- the modification is larger than the one declared'], ...
            m.file, hRebuilt, files(k).sha256_imported); %#ok<AGROW>
        continue
    end

    report.patch_checks{end+1} = sprintf( ...
        ['%s: audited source (%s) + %s reproduces the running file (%s) byte ' ...
         'for byte -- %s'], m.file, shortHash(hVerbatim), m.diff, ...
        shortHash(hRebuilt), m.kind); %#ok<AGROW>
end

% ---- PROVENANCE ONLY: the state of the external source repository -------
% Measured and recorded; never blocking.  See the header.
report.source_repository_state = sourceRepositoryState(man, files, declaredDest);
report.source_repository = man.source_repository.path;      % legacy field name
report.source_reachable  = report.source_repository_state.reachable;

% ---- Dispatch gate -------------------------------------------------------
try
    [g, resolved] = olhoffm4_paths(); %#ok<ASGLU>
    report.dispatch_ok = true;
    report.resolved = resolved;
    clear g
catch ME
    report.dispatch_ok = false;
    report.dispatch_error = ME.message;
    report.resolved = struct('name', {}, 'file', {});
end

report.pass = isempty(report.imported_hash_mismatches) && ...
              isempty(report.unattested_files) && ...
              isempty(report.declared_patch_mismatches) && ...
              report.dispatch_ok;

if verbose
    s = report.source_repository_state;
    fprintf('  imported files hashed        : %d\n', report.checked);
    fprintf('  integrity mismatches         : %d\n', numel(report.imported_hash_mismatches));
    fprintf('  unattested files             : %d\n', numel(report.unattested_files));
    fprintf('  declared modification(s)     : %s\n', strjoin(report.declared_modifications, ', '));
    fprintf('  patch reconstruction failures: %d\n', numel(report.declared_patch_mismatches));
    fprintf('  dispatch gate                : %s\n', ternary(report.dispatch_ok, 'PASS', 'FAIL'));
    fprintf('  external source (provenance) : %s\n', s.summary);
    for i = 1:numel(report.patch_checks)
        fprintf('    . %s\n', report.patch_checks{i});
    end
    for i = 1:numel(report.imported_hash_mismatches)
        fprintf('    ! INTEGRITY: %s\n', report.imported_hash_mismatches{i});
    end
    for i = 1:numel(report.unattested_files)
        fprintf('    ! UNATTESTED: %s\n', report.unattested_files{i});
    end
    for i = 1:numel(report.declared_patch_mismatches)
        fprintf('    ! RECONSTRUCTION: %s\n', report.declared_patch_mismatches{i});
    end
end
end

% =========================================================================
function s = sourceRepositoryState(man, files, declaredDest)
%SOURCEREPOSITORYSTATE  Measure the external source directory.  Never blocking.
srcRoot = man.source_repository.path;
s = struct();
s.path = srcRoot;
s.role = ['PROVENANCE ONLY. This directory is unversioned and outside this ' ...
    'repository. Its present state is recorded, and is deliberately not part ' ...
    'of the pass/fail decision: the running solver is pinned to the audited ' ...
    'source by in-repository evidence (integrity + attestation + patch ' ...
    'reconstruction), which a change here cannot affect.'];
s.reachable = exist(srcRoot, 'dir') == 7;
s.files_checked = 0;
s.files_matching = 0;
s.files_absent = {};
s.files_differing = {};
s.differs_from_import_without_declaration = {};

if s.reachable
    for i = 1:numel(files)
        e = files(i);
        src = fullfile(srcRoot, e.source_path);
        if exist(src, 'file') ~= 2
            s.files_absent{end+1} = e.source_path; %#ok<AGROW>
            continue
        end
        s.files_checked = s.files_checked + 1;
        hSrc = olhoffm4_sha256_file(src);
        if strcmp(hSrc, e.sha256_source)
            s.files_matching = s.files_matching + 1;
        else
            s.files_differing{end+1} = sprintf('%s (%s, recorded %s)', ...
                e.source_path, shortHash(hSrc), shortHash(e.sha256_source)); %#ok<AGROW>
        end
        if ~strcmp(hSrc, e.sha256_imported) && ~any(strcmp(e.destination_path, declaredDest))
            s.differs_from_import_without_declaration{end+1} = e.source_path; %#ok<AGROW>
        end
    end
end

s.in_imported_state = s.reachable && isempty(s.files_absent) && isempty(s.files_differing);
if ~s.reachable
    s.summary = sprintf('%s is not reachable; provenance rests on the in-repository evidence', srcRoot);
elseif s.in_imported_state
    s.summary = sprintf('%d of %d audited files still in the imported state', ...
        s.files_matching, numel(files));
else
    s.summary = sprintf(['NO LONGER IN THE IMPORTED STATE: %d file(s) absent, ' ...
        '%d differing, %d unchanged (of %d). Not blocking; the running solver ' ...
        'is pinned by in-repository evidence.'], numel(s.files_absent), ...
        numel(s.files_differing), s.files_matching, numel(files));
end
end

function h = shortHash(h)
if ischar(h) && numel(h) > 12; h = [h(1:12) '...']; end
end

function s = ternary(tf, a, b)
if tf; s = a; else; s = b; end
end
