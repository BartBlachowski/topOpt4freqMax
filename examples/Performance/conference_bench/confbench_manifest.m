function man = confbench_manifest(cfg, methodConfigs, resolvedImpl)
%CONFBENCH_MANIFEST  Record exactly what is about to run.  Output, not control.
%
%   man = CONFBENCH_MANIFEST(cfg, methodConfigs, resolvedImpl)
%
%   The manifest is GENERATED FROM the runtime configuration; it never feeds
%   back into it.  The direction is
%
%       user edits cfg  ->  cfg validated  ->  benchmark runs  ->  manifest records
%
%   and never the reverse.  Editing this file changes what is RECORDED, not
%   what is RUN; to change what runs, edit cfg.resolutions in
%   performance_comparison.m.
%
%   See also CONFBENCH_PREFLIGHT, PERFORMANCE_COMPARISON.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));

man = struct();
man.manifest_schema = 'conference_performance_benchmark/1';
man.generated_datetime = datetime('now', 'TimeZone', 'local', 'Format', ...
    'yyyy-MM-dd''T''HH:mm:ssXXX');
man.generated_datetime = char(string(man.generated_datetime));
man.generated_by = 'examples/Performance/performance_comparison.m';
man.manifest_role = ['OUTPUT. Generated from the runtime configuration. It is ' ...
    'not a source of scientific settings and nothing reads it back.'];

% ---- the runtime configuration, verbatim -------------------------------
man.configuration = cfg;
man.active_resolutions = cfg.resolutions;
man.active_resolutions_count = size(cfg.resolutions, 1);
man.active_methods = activeMethodList(cfg);
man.element_counts = cfg.resolutions(:,1) .* cfg.resolutions(:,2);

% ---- per-method frozen scientific settings -----------------------------
man.method_configurations = methodConfigs;

% ---- implementation identity -------------------------------------------
man.resolved_implementations = resolvedImpl;

% ---- source hashes ------------------------------------------------------
tracked = { ...
    'examples/Performance/performance_comparison.m', ...
    'examples/Performance/conference_bench/confbench_method_config.m', ...
    'examples/Performance/conference_bench/confbench_run_case.m', ...
    'examples/Performance/conference_bench/confbench_preflight.m', ...
    'examples/Performance/conference_bench/confbench_manifest.m', ...
    'examples/Performance/conference_bench/confbench_export.m', ...
    'examples/Performance/conference_bench/confbench_timing_schema.m', ...
    'examples/Performance/conference_bench/confbench_caveats.m', ...
    'examples/Performance/conference_bench/confbench_scaling_fit.m', ...
    'analysis/OlhoffM4Reconstruction/olhoffm4_config.m', ...
    'analysis/OlhoffM4Reconstruction/olhoffm4_run.m', ...
    'analysis/OlhoffM4Reconstruction/olhoffm4_paths.m', ...
    'analysis/ourApproach/Matlab/topopt_freq.m', ...
    'analysis/YukselApproach/Matlab/top99neo_inertial_freq.m', ...
    'tools/Matlab/run_topopt_from_json.m', ...
    'analysis/three_method_parametric_study/study_base_config.m', ...
    'analysis/three_method_parametric_study/study_evaluate_design.m', ...
    'analysis/three_method_parametric_study/results/profile_freeze_manifest.json'};
man.source_hashes = struct();
for i = 1:numel(tracked)
    p = fullfile(repo, tracked{i});
    key = matlab.lang.makeValidName(tracked{i});
    if exist(p, 'file') == 2
        man.source_hashes.(key) = struct('path', tracked{i}, ...
            'sha256', olhoffm4_sha256_file(p));
    else
        man.source_hashes.(key) = struct('path', tracked{i}, 'sha256', 'MISSING');
    end
end

% ---- imported Olhoff hashes, re-read at run time ------------------------
importManifestPath = fullfile(repo, 'analysis', 'OlhoffM4Reconstruction', 'IMPORT_MANIFEST.json');
imp = jsondecode(fileread(importManifestPath));
man.olhoff_import = struct( ...
    'namespace', 'analysis/OlhoffM4Reconstruction', ...
    'import_manifest', 'analysis/OlhoffM4Reconstruction/IMPORT_MANIFEST.json', ...
    'import_manifest_sha256', olhoffm4_sha256_file(importManifestPath), ...
    'source_repository', imp.source_repository.path, ...
    'import_datetime', imp.import_datetime_local, ...
    'epistemic_status', imp.epistemic_status, ...
    'declared_modifications', {{imp.modifications.file}}, ...
    'files', struct('destination_path', {}, 'sha256_now', {}, 'sha256_at_import', {}, 'match', {}));
for i = 1:numel(imp.files)
    e = imp.files(i);
    p = fullfile(repo, e.destination_path);
    hNow = 'MISSING';
    if exist(p, 'file') == 2; hNow = olhoffm4_sha256_file(p); end
    man.olhoff_import.files(end+1) = struct( ...
        'destination_path', e.destination_path, ...
        'sha256_now', hNow, ...
        'sha256_at_import', e.sha256_imported, ...
        'match', strcmp(hNow, e.sha256_imported)); %#ok<AGROW>
end
man.olhoff_import.all_files_match = all([man.olhoff_import.files.match]);

% ---- environment --------------------------------------------------------
man.environment = struct( ...
    'matlab_version', version(), ...
    'matlab_release', version('-release'), ...
    'computer', computer(), ...
    'max_num_comp_threads', maxNumCompThreads(), ...
    'single_thread_requested', cfg.singleThread, ...
    'blas_library', blasLibraryOrUnknown(), ...
    'hostname', hostnameOrUnknown());

% ---- repository state ---------------------------------------------------
man.repository = repositoryState(repo);

% ---- what this run is, epistemically -----------------------------------
man.run_class = struct( ...
    'run_label', cfg.runLabel, ...
    'scientific_evidence', cfg.scientificEvidence, ...
    'performance_campaign', cfg.performanceCampaign, ...
    'max_outer_override', cfg.maxOuterOverride, ...
    'warmup_performed', cfg.runWarmup);

man.caveats = confbench_caveats();
man.timing_schema = confbench_timing_schema();
end

% =========================================================================
function names = activeMethodList(cfg)
names = {};
fn = fieldnames(cfg.methods);
for i = 1:numel(fn)
    if cfg.methods.(fn{i}); names{end+1} = fn{i}; end %#ok<AGROW>
end
end

function s = repositoryState(repo)
s = struct('path', repo, 'git_available', false, 'head', 'UNKNOWN', ...
    'branch', 'UNKNOWN', 'dirty', 'UNKNOWN', 'dirty_files', {{}});
[st, head] = system(sprintf('git -C "%s" rev-parse HEAD', repo));
if st ~= 0; return; end
s.git_available = true;
s.head = strtrim(head);
[~, br] = system(sprintf('git -C "%s" rev-parse --abbrev-ref HEAD', repo));
s.branch = strtrim(br);
[~, porcelain] = system(sprintf('git -C "%s" status --porcelain', repo));
porcelain = strtrim(porcelain);
s.dirty = ~isempty(porcelain);
if s.dirty
    lines = strsplit(porcelain, newline);
    s.dirty_files = lines(~cellfun(@isempty, lines));
    s.dirty_file_count = numel(s.dirty_files);
else
    s.dirty_file_count = 0;
end
end

function v = blasLibraryOrUnknown()
try
    v = version('-blas');
catch
    v = 'UNKNOWN';
end
end

function h = hostnameOrUnknown()
try
    [st, h] = system('hostname');
    if st ~= 0; h = 'UNKNOWN'; else; h = strtrim(h); end
catch
    h = 'UNKNOWN';
end
end
