clear; clc;

% Determinism precondition for the controlled benchmark, section 6.1 of
% examples/Performance/PLAN_two_table_redesign.md.
%
% Offline acceptance, timed replay to a discovered k*, and the paired prefix
% invariance test all compare two separate executions of the same solver.  None
% of them means anything unless the solvers are bit-reproducible.  MATLAB's EIGS
% draws its start vector from the global random stream when none is supplied,
% which makes a trajectory depend on stream state and therefore on the order in
% which runs execute within a session -- exactly what the timing protocol's
% run-order randomization varies.
%
% This script establishes three properties per (mesh, method):
%
%   1. repeated runs are bit-identical;
%   2. results do not depend on the global random-stream state;
%   3. results do not depend on the order methods execute within a session.
%
% It also checks directly that no solver consumes the global stream.

thisDir = fileparts(mfilename('fullpath'));
baseCfg = jsondecode(fileread(fullfile(thisDir, 'performance_comparison.json')));
baseCfg.postprocessing.visualize_live = false;
baseCfg.postprocessing.save_final_image = false;
baseCfg.postprocessing.save_snapshot_image = false;
baseCfg.postprocessing.save_frequency_iterations = false;
baseCfg.optimization.filter.radius = 2;
baseCfg.optimization.filter.radius_units = 'element';

meshes     = [160, 20; 320, 40];
approaches = {'Olhoff',         'Yuksel',         'OurApproach'};
labels     = {'OlhoffApproach', 'YukselApproach', 'ProposedApproach'};
nMethods   = numel(approaches);

% Pass 1 is the reference.  Pass 2 repeats it from a different global stream
% state; pass 3 also reverses the execution order.  A solver that draws from the
% global stream cannot match the reference in both.
passes = struct( ...
    'name',  {'forward, seed 1', 'forward, seed 2', 'reverse, seed 3'}, ...
    'seed',  {1, 2, 3}, ...
    'order', {1:nMethods, 1:nMethods, nMethods:-1:1});
nPasses = numel(passes);

records  = cell(size(meshes,1), nMethods, nPasses);
failures = {};

for p = 1:nPasses
    rng(passes(p).seed, 'twister');
    for r = 1:size(meshes,1)
        for m = passes(p).order
            cfg = baseCfg;
            cfg.domain.mesh.nelx = meshes(r,1);
            cfg.domain.mesh.nely = meshes(r,2);
            cfg.optimization.approach = approaches{m};

            fprintf('pass %d (%s)  %-18s  %dx%d ...\n', p, passes(p).name, ...
                labels{m}, meshes(r,1), meshes(r,2));

            streamBefore = rng;
            [x, omega, ~, nIter, ~, nIterStage, telemetry] = run_topopt_from_json(cfg);
            streamAfter = rng;

            rec = struct();
            rec.x                 = x(:);
            rec.omega             = omega(:);
            rec.n_iter            = nIter;
            rec.stage1            = nIterStage.stage1;
            rec.stage2            = nIterStage.stage2;
            rec.objective_history = telemetry.objective_history(:);
            rec.objective_final   = telemetry.objective_final;
            rec.stream_untouched  = isequal(streamBefore.State, streamAfter.State) ...
                && strcmp(streamBefore.Type, streamAfter.Type);
            records{r, m, p} = rec;

            if ~rec.stream_untouched
                failures{end+1} = sprintf(['%s %dx%d pass %d consumed the global ' ...
                    'random stream'], labels{m}, meshes(r,1), meshes(r,2), p); %#ok<SAGROW>
            end
        end
    end
end

% ------------------------------------------------------------------------
% Compare every pass against the pass-1 reference
% ------------------------------------------------------------------------
sepWidth = 104;
sep = repmat('-', 1, sepWidth);

fprintf('\nDeterminism validation, section 6.1\n');
fprintf('%s\n', sep);
fprintf('%-18s %-9s %-18s %8s %14s %14s %14s\n', ...
    'Method', 'Mesh', 'Comparison', 'iters', 'max|dx|', 'max|domega|', 'max|dobj|');
fprintf('%s\n', sep);

comparisons = struct('method', {}, 'mesh', {}, 'comparison', {}, ...
    'iterations_reference', {}, 'iterations_other', {}, ...
    'topology_max_abs_difference', {}, 'frequency_max_abs_difference', {}, ...
    'objective_history_max_abs_difference', {}, 'bit_identical', {}, ...
    'global_stream_untouched', {});

for r = 1:size(meshes,1)
    meshStr = sprintf('%dx%d', meshes(r,1), meshes(r,2));
    for m = 1:nMethods
        ref = records{r, m, 1};
        for p = 2:nPasses
            other = records{r, m, p};

            dx = max(abs(ref.x - other.x));
            dw = max(abs(ref.omega - other.omega), [], 'omitnan');
            if numel(ref.objective_history) == numel(other.objective_history)
                dobj = max(abs(ref.objective_history - other.objective_history), [], 'omitnan');
            else
                dobj = NaN;   % differing lengths are already a failure below
            end
            if isempty(dobj)
                % Some solvers report no objective history; the density,
                % frequency, and iteration comparisons below still apply.
                dobj = NaN;
            end

            bitIdentical = isequaln(ref.x, other.x) ...
                && isequaln(ref.omega, other.omega) ...
                && isequaln(ref.n_iter, other.n_iter) ...
                && isequaln(ref.stage1, other.stage1) ...
                && isequaln(ref.stage2, other.stage2) ...
                && isequaln(ref.objective_history, other.objective_history);

            if ~bitIdentical
                failures{end+1} = sprintf(['%s %s: pass %d differs from reference ' ...
                    '(iters %d vs %d, max|dx|=%.3e)'], labels{m}, meshStr, p, ...
                    ref.n_iter, other.n_iter, dx); %#ok<SAGROW>
            end

            fprintf('%-18s %-9s %-18s %8s %14.3e %14.3e %14.3e\n', ...
                labels{m}, meshStr, sprintf('pass 1 vs %d', p), ...
                sprintf('%d/%d', ref.n_iter, other.n_iter), dx, dw, dobj);

            comparisons(end+1) = struct( ...
                'method', labels{m}, ...
                'mesh', meshStr, ...
                'comparison', sprintf('pass 1 (%s) vs pass %d (%s)', ...
                    passes(1).name, p, passes(p).name), ...
                'iterations_reference', ref.n_iter, ...
                'iterations_other', other.n_iter, ...
                'topology_max_abs_difference', dx, ...
                'frequency_max_abs_difference', dw, ...
                'objective_history_max_abs_difference', dobj, ...
                'bit_identical', bitIdentical, ...
                'global_stream_untouched', ref.stream_untouched && other.stream_untouched); %#ok<SAGROW>
        end
    end
end
fprintf('%s\n', sep);

passed = isempty(failures);
if passed
    fprintf(['PASS: every method is bit-identical across repetition, global stream ' ...
        'state, and execution order,\n      and no solver consumed the global stream.\n']);
else
    fprintf('FAIL: %d determinism violation(s):\n', numel(failures));
    for k = 1:numel(failures)
        fprintf('  - %s\n', failures{k});
    end
end

validation = struct();
validation.description = ['Determinism precondition, section 6.1 of ' ...
    'examples/Performance/PLAN_two_table_redesign.md'];
validation.eigs_start_vector_policy = ['Fixed unit-norm vector from a private ' ...
    'RandStream(''twister'', ''Seed'', 42); the global stream is not used.'];
validation.meshes = arrayfun(@(k) sprintf('%dx%d', meshes(k,1), meshes(k,2)), ...
    (1:size(meshes,1))', 'UniformOutput', false);
validation.methods = labels(:);
passInfo = cell(nPasses, 1);
for k = 1:nPasses
    % The extra brace keeps the method-order cell array as one field value
    % instead of expanding it into a struct array.
    passInfo{k} = struct('name', passes(k).name, 'seed', passes(k).seed, ...
        'order', {labels(passes(k).order)});
end
validation.passes = passInfo;
validation.comparisons = comparisons;
validation.failures = failures(:);
validation.pass = passed;

validationPath = fullfile(thisDir, 'determinism_validation.json');
fid = fopen(validationPath, 'w');
assert(fid >= 0, 'validate_determinism:OpenFailed', ...
    'Cannot open %s for writing.', validationPath);
fprintf(fid, '%s\n', jsonencode(validation));
fclose(fid);
fprintf('Validation record saved to: %s\n', validationPath);

assert(passed, 'validate_determinism:NotDeterministic', ...
    'Solvers are not bit-reproducible; offline acceptance and timed replay are invalid.');
