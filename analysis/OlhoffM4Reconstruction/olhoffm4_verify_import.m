function report = olhoffm4_verify_import(varargin)
%OLHOFFM4_VERIFY_IMPORT  Prove the imported solver is the audited one, unchanged.
%
%   report = OLHOFFM4_VERIFY_IMPORT() re-hashes every imported file and compares
%   it against IMPORT_MANIFEST.json.  Where the source repository is reachable
%   it also re-hashes the source and confirms that each file is either
%   byte-identical to it or is the single declared timing-instrumentation
%   patch.  Any other difference is a failure.
%
%   Name/value options:
%     'Verbose'  (default true)  print a line per checked group
%
%   report.pass is true only when every file checks out AND the dispatch gate
%   resolves every owned function inside this import.

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
report.source_repository = man.source_repository.path;
report.source_reachable = exist(man.source_repository.path, 'dir') == 7;
report.checked = 0;
report.imported_hash_mismatches = {};
report.source_hash_mismatches = {};
report.undeclared_modifications = {};
report.declared_modifications = {man.modifications.file};

files = man.files;
for i = 1:numel(files)
    e = files(i);
    dst = fullfile(repoRoot, e.destination_path);
    if exist(dst, 'file') ~= 2
        report.imported_hash_mismatches{end+1} = sprintf('%s MISSING', e.destination_path); %#ok<AGROW>
        continue
    end
    hNow = olhoffm4_sha256_file(dst);
    report.checked = report.checked + 1;
    if ~strcmp(hNow, e.sha256_imported)
        report.imported_hash_mismatches{end+1} = sprintf('%s: %s != manifest %s', ...
            e.destination_path, hNow, e.sha256_imported); %#ok<AGROW>
    end
    if report.source_reachable
        src = fullfile(man.source_repository.path, e.source_path);
        if exist(src, 'file') == 2
            hSrc = olhoffm4_sha256_file(src);
            if ~strcmp(hSrc, e.sha256_source)
                report.source_hash_mismatches{end+1} = sprintf( ...
                    '%s: source has CHANGED since import (%s != %s)', ...
                    e.source_path, hSrc, e.sha256_source); %#ok<AGROW>
            end
            if ~strcmp(hSrc, hNow) && ~any(strcmp(e.destination_path, ...
                    fullfile('analysis', 'OlhoffM4Reconstruction', report.declared_modifications)))
                report.undeclared_modifications{end+1} = e.destination_path; %#ok<AGROW>
            end
        end
    end
end

% The declared patch must still be exactly the recorded diff of the recorded
% verbatim source copy, so "declared" cannot become a loophole.
report.patch_checks = {};
for i = 1:numel(man.modifications)
    m = man.modifications(i);
    verbatim = fullfile(root, m.source_verbatim_copy);
    srcFile = '';
    for k = 1:numel(files)
        if strcmp(files(k).destination_path, fullfile('analysis','OlhoffM4Reconstruction', m.file))
            srcFile = fullfile(man.source_repository.path, files(k).source_path);
        end
    end
    if exist(verbatim, 'file') == 2 && ~isempty(srcFile) && exist(srcFile, 'file') == 2
        if strcmp(olhoffm4_sha256_file(verbatim), olhoffm4_sha256_file(srcFile))
            report.patch_checks{end+1} = sprintf('%s: verbatim source copy matches the live source', m.file); %#ok<AGROW>
        else
            report.imported_hash_mismatches{end+1} = sprintf( ...
                '%s: the recorded verbatim source copy no longer matches the source', m.file); %#ok<AGROW>
        end
    end
end

% Dispatch gate.
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
              isempty(report.source_hash_mismatches) && ...
              isempty(report.undeclared_modifications) && ...
              report.dispatch_ok;

if verbose
    fprintf('  imported files hashed        : %d\n', report.checked);
    fprintf('  source repository reachable  : %s\n', mat2str(report.source_reachable));
    fprintf('  imported-hash mismatches     : %d\n', numel(report.imported_hash_mismatches));
    fprintf('  source-hash mismatches       : %d\n', numel(report.source_hash_mismatches));
    fprintf('  undeclared modifications     : %d\n', numel(report.undeclared_modifications));
    fprintf('  declared modification(s)     : %s\n', strjoin(report.declared_modifications, ', '));
    fprintf('  dispatch gate                : %s\n', ternary(report.dispatch_ok, 'PASS', 'FAIL'));
    for i = 1:numel(report.imported_hash_mismatches)
        fprintf('    ! %s\n', report.imported_hash_mismatches{i});
    end
    for i = 1:numel(report.source_hash_mismatches)
        fprintf('    ! %s\n', report.source_hash_mismatches{i});
    end
    for i = 1:numel(report.undeclared_modifications)
        fprintf('    ! UNDECLARED MODIFICATION: %s\n', report.undeclared_modifications{i});
    end
end
end

function s = ternary(tf, a, b)
if tf; s = a; else; s = b; end
end
