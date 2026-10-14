function nFail = test_stop_strategies_unchanged(varargin)
%TEST_STOP_STRATEGIES_UNCHANGED  The pre-existing stop strategies are bitwise unchanged.
%
%   nFail = TEST_STOP_STRATEGIES_UNCHANGED() runs every stop strategy that
%   existed before 'stagnation_x' was added, for each method at 160x20, through
%   the benchmark configuration path (confbench_method_config), and compares
%   with the digests recorded BEFORE the change in
%   fixtures/stop_strategies_pre_stagnation_x.json:
%     final design x (bytes), omega (bytes), iteration counts (Yuksel per
%     stage), and every stopping field that existed then (stop reason,
%     criterion, final change measures, window values ...); for Du-Olhoff also
%     the solver log.  Fields added later are not compared.
%
%   Cases (the same list recorded the fixture):
%     Proposed   max_change (frozen profile), relative_l2_change 1e-2,
%                stagnation (defaults), stagnation 10/1e-2/1e-2
%     Yuksel     max_change, relative_l2_change 1e-2 both stages, stagnation
%                both stages, stagnation with stage 1 on max_change 0.01
%     Du-Olhoff  l2_change (default), max_change 0.03, relative_l2_change 1e-3,
%                stagnation (defaults); outer cap 120
%
%   The four Proposed records were re-recorded on 2026-10-14 after the
%   fairness audit changed its assembly and solve (lower-triangle assembly,
%   Cholesky of the lower triangle, objective u'f): same iteration counts and
%   stop reasons, designs within 6e-11, omega_1 within 6e-12 relative
%   (examples/Performance/conference_bench/FAIRNESS_AUDIT_2026-10-14.md).
%   Yuksel and Du-Olhoff records are the originals and still match bitwise.
%
%   HOST-DEPENDENT.  Bitwise digests hold on the recording host only (macOS
%   arm64, MATLAB R2025b, maxNumCompThreads = 1, each method in a fresh
%   session); another BLAS gives other last bits (and Du-Olhoff 160x20 other
%   iteration counts).  Run each method in a FRESH MATLAB session: Du-Olhoff at
%   160x20 depends on session history (see performance_comparison runWarmup).
%
%   Options:
%     'Methods'   cellstr, default {'proposed','yuksel','olhoff'}
%     'WriteFixtureFrom'  cellstr of .mat files holding R, recorded by the same
%                 case list before the change; writes the fixture and returns.
%                 Used once, when the fixture was created.

p = inputParser();
p.addParameter('Methods', {'proposed', 'yuksel', 'olhoff'}, @iscellstr);
p.addParameter('WriteFixtureFrom', {}, @iscellstr);
p.parse(varargin{:});
opt = p.Results;

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
fixtureFile = fullfile(here, 'fixtures', 'stop_strategies_pre_stagnation_x.json');

if ~isempty(opt.WriteFixtureFrom)
    F = struct('source', ['digests of the pre-existing stop strategies, recorded before ' ...
        'stagnation_x was added; host macOS arm64, MATLAB R2025b, single thread'], ...
        'cases', {{}});
    for i = 1:numel(opt.WriteFixtureFrom)
        L = load(opt.WriteFixtureFrom{i}, 'R', 'method');
        for k = 1:numel(L.R)
            F.cases{end+1} = digestCase(L.method, L.R(k), {}); %#ok<AGROW>
        end
    end
    fid = fopen(fixtureFile, 'w');
    fprintf(fid, '%s\n', jsonencode(F, 'PrettyPrint', true));
    fclose(fid);
    nFail = 0;
    fprintf('wrote %s (%d cases)\n', fixtureFile, numel(F.cases));
    return
end

addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'examples', 'Performance', 'conference_bench'));
addpath(fullfile(repo, 'examples', 'Performance', 'benchmark_profile'));
addpath(fullfile(repo, 'analysis', 'Olhoff'));
olhoffcurrent_scrub_forbidden_paths(repo);
entryThreads = maxNumCompThreads(1);
restoreThreads = onCleanup(@() maxNumCompThreads(entryThreads));

F = jsondecode(fileread(fixtureFile));
if isstruct(F.cases); F.cases = num2cell(F.cases); end
nFail = 0;
fprintf('\n%s\nTEST_STOP_STRATEGIES_UNCHANGED\n%s\n', repmat('=',1,72), repmat('=',1,72));
for m = 1:numel(opt.Methods)
    method = opt.Methods{m};
    R = runCases(method);
    for k = 1:numel(R)
        ref = findCase(F.cases, method, R(k).name);
        got = digestCase(method, R(k), ref.stopping_fields);
        ok = true; why = {};
        for f = {'n_iter', 'n_stage1', 'n_stage2', 'x_sha256', 'omega_sha256', 'stopping_sha256', 'log_sha256'}
            if ~isequal(got.(f{1}), ref.(f{1}))
                ok = false; why{end+1} = f{1}; %#ok<AGROW>
            end
        end
        label = sprintf('%s %-20s nIter %d', method, R(k).name, got.n_iter);
        if ~ok, label = [label ' -- differs: ' strjoin(why, ', ')]; end %#ok<AGROW>
        nFail = nFail + chk(label, ok);
    end
end
fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

% =========================================================================
function R = runCases(method)
nelx = 160; nely = 20;
switch method
    case 'proposed'
        cases = {
            'default_max_change', struct()
            'relative_l2_1e-2',   struct('criterion','relative_l2_change','tol',1e-2)
            'stagnation_default', struct('criterion','stagnation')
            'stagnation_1e-2',    struct('criterion','stagnation','window',10,'objectiveTol',1e-2,'graynessTol',1e-2)
            };
    case 'yuksel'
        cases = {
            'default_max_change', struct()
            'relative_l2_1e-2',   struct('criterion','relative_l2_change','stage1Tol',1e-2,'stage2Tol',1e-2)
            'stagnation_default', struct('criterion','stagnation')
            'stagnation_s1max',   struct('criterion','stagnation','stage1Criterion','max_change','stage1Tol',0.01)
            };
    case 'olhoff'
        cases = {
            'default_l2_change',  struct('maxOuter',120)
            'max_change_0.03',    struct('criterion','max_change','tol',0.03,'maxOuter',120)
            'relative_l2_1e-3',   struct('criterion','relative_l2_change','tol',1e-3,'maxOuter',120)
            'stagnation_default', struct('criterion','stagnation','maxOuter',120)
            };
end
R = struct('name', {}, 'x', {}, 'omega', {}, 'nIter', {}, 'stage', {}, 'stopping', {}, 'log', {});
for k = 1:size(cases, 1)
    mcfg = confbench_method_config(method, nelx, nely, '', cases{k,2});
    if strcmp(method, 'olhoff')
        args = {'Preset', mcfg.olhoff_preset};
        if ~isempty(mcfg.olhoff_stop_factor); args = [args, {'StopToleranceFactor', mcfg.olhoff_stop_factor}]; end %#ok<AGROW>
        if ~isempty(mcfg.olhoff_stop_max_change_tol); args = [args, {'StopMaxChangeTolerance', mcfg.olhoff_stop_max_change_tol}]; end %#ok<AGROW>
        if ~isempty(mcfg.olhoff_stop_relative_tol); args = [args, {'StopRelativeChangeTolerance', mcfg.olhoff_stop_relative_tol}]; end %#ok<AGROW>
        if ~isempty(mcfg.olhoff_stop_stagnation); args = [args, {'StopStagnation', mcfg.olhoff_stop_stagnation}]; end %#ok<AGROW>
        if ~isempty(mcfg.olhoff_max_outer); args = [args, {'MaxOuter', mcfg.olhoff_max_outer}]; end %#ok<AGROW>
        o = olhoffcurrent_run(nelx, nely, args{:});
        R(end+1) = struct('name', cases{k,1}, 'x', o.x, 'omega', o.omega, ...
            'nIter', o.accounting.outer_iterations, 'stage', [], 'stopping', o.stopping, ...
            'log', {o.log}); %#ok<AGROW>
    else
        [x, omega, ~, nIter, ~, nIterStage, tel] = run_topopt_from_json(mcfg);
        R(end+1) = struct('name', cases{k,1}, 'x', x, 'omega', omega, 'nIter', nIter, ...
            'stage', nIterStage, 'stopping', tel.stopping, 'log', {{}}); %#ok<AGROW>
    end
end
end

function D = digestCase(method, r, fields)
% fields: the stopping fields to digest; {} = every field present (recording).
s = r.stopping;
if isempty(fields)
    fields = fieldnames(s);
    % Wall-clock and host-specific fields never enter a digest.
    fields = fields(~contains(fields, {'time', 'elapsed', 'wall'}));
end
fields = fields(:).';
parts = cell(1, numel(fields));
for i = 1:numel(fields)
    parts{i} = sprintf('%s=%s', fields{i}, show(s.(fields{i})));
end
n1 = NaN; n2 = NaN;
if isstruct(r.stage), n1 = r.stage.stage1; n2 = r.stage.stage2; end
D = struct('method', method, 'name', r.name, 'n_iter', double(r.nIter), ...
    'n_stage1', nanToStr(n1), 'n_stage2', nanToStr(n2), ...
    'x_sha256', sha(typecast(double(r.x(:)).', 'uint8')), ...
    'omega_sha256', sha(typecast(double(r.omega(:)).', 'uint8')), ...
    'stopping_fields', {fields}, ...
    'stopping_sha256', sha(uint8(strjoin(parts, newline))), ...
    'log_sha256', sha(uint8(strjoin(cellstr(r.log), newline))));
end

function v = nanToStr(n)
% JSON has no NaN; a missing stage count is recorded as the string "NaN".
if isnan(n), v = 'NaN'; else, v = double(n); end
end

function c = findCase(cases, method, name)
for i = 1:numel(cases)
    if strcmp(cases{i}.method, method) && strcmp(cases{i}.name, name), c = cases{i}; return; end
end
error('test_stop_strategies_unchanged:NoFixture', 'No recorded case %s/%s.', method, name);
end

function h = sha(bytes)
md = java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(bytes), md.update(bytes(:)); end   % an empty input hashes as ''
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end

function s = show(v)
if ischar(v);            s = v;
elseif isstring(v);      s = char(v);
elseif islogical(v);     s = mat2str(v);
elseif isnumeric(v);     s = mat2str(double(v), 17);
elseif iscell(v);        s = ['{' strjoin(cellfun(@show, v, 'UniformOutput', false), ',') '}'];
elseif isstruct(v);      s = jsonencode(v);
else,                    s = class(v);
end
end

function n = chk(label, ok)
if ok, fprintf('  [PASS] %s\n', label); n = 0;
else,  fprintf('  [FAIL] %s\n', label); n = 1; end
end
