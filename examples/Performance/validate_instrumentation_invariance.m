clear; clc;

% Representative benchmark case used to verify that disabling development
% diagnostics changes instrumentation only, never the numerical trajectory.
thisDir = fileparts(mfilename('fullpath'));
cfg = jsondecode(fileread(fullfile(thisDir, 'performance_comparison.json')));
cfg.domain.mesh.nelx = 160;
cfg.domain.mesh.nely = 20;
cfg.optimization.approach = 'OurApproach';
cfg.postprocessing.visualize_live = false;
cfg.postprocessing.save_final_image = false;
cfg.postprocessing.save_snapshot_image = false;
cfg.postprocessing.save_frequency_iterations = false;

legacyCfg = cfg;
legacyCfg.benchmark.enable_diagnostics = true;
legacyTic = tic;
[xBefore, omegaBefore, tIterBefore, nIterBefore, ~, ~, telemetryBefore] = ...
    run_topopt_from_json(legacyCfg);
wallBefore = toc(legacyTic);

benchmarkCfg = cfg;
benchmarkCfg.benchmark.enable_diagnostics = false;
benchmarkTic = tic;
[xAfter, omegaAfter, tIterAfter, nIterAfter, ~, ~, telemetryAfter] = ...
    run_topopt_from_json(benchmarkCfg);
wallAfter = toc(benchmarkTic);

topologyMaxAbsDiff = max(abs(xBefore - xAfter));
frequencyMaxAbsDiff = max(abs(omegaBefore - omegaAfter), [], 'omitnan');
objectiveBefore = telemetryBefore.objective_history;
objectiveAfter = telemetryAfter.objective_history;
assert(numel(objectiveBefore) == numel(objectiveAfter), ...
    'validate_instrumentation_invariance:ObjectiveHistoryLength', ...
    'Objective-history lengths differ.');
objectiveMaxAbsDiff = max(abs(objectiveBefore - objectiveAfter), [], 'omitnan');

assert(nIterBefore == nIterAfter, ...
    'validate_instrumentation_invariance:IterationMismatch', ...
    'Iteration counts differ with diagnostics on/off.');
assert(isequaln(xBefore, xAfter), ...
    'validate_instrumentation_invariance:TopologyMismatch', ...
    'Final topology differs with diagnostics on/off.');
assert(isequaln(objectiveBefore, objectiveAfter), ...
    'validate_instrumentation_invariance:ObjectiveHistoryMismatch', ...
    'Objective history differs with diagnostics on/off.');
assert(isequaln(omegaBefore, omegaAfter), ...
    'validate_instrumentation_invariance:FrequencyMismatch', ...
    'Final frequencies differ with diagnostics on/off.');

fprintf('\nInstrumentation invariance validation: Proposed, 160x20\n');
fprintf('  iterations before/after: %d / %d\n', nIterBefore, nIterAfter);
fprintf('  final objective before/after: %.17g / %.17g\n', ...
    telemetryBefore.objective_final, telemetryAfter.objective_final);
fprintf('  objective-history max |difference|: %.3e\n', objectiveMaxAbsDiff);
fprintf('  final-frequency max |difference|: %.3e rad/s\n', frequencyMaxAbsDiff);
fprintf('  topology max |difference|: %.3e\n', topologyMaxAbsDiff);
fprintf('  topology checksum before: %s\n', numeric_fingerprint(xBefore));
fprintf('  topology checksum after : %s\n', numeric_fingerprint(xAfter));
fprintf('  legacy reconstructed runtime before/after: %.6f / %.6f s\n', ...
    tIterBefore*nIterBefore, tIterAfter*nIterAfter);
fprintf('  measured wall runtime before/after: %.6f / %.6f s\n', wallBefore, wallAfter);
fprintf('  after timing fields: init=%.6f, loop=%.6f, post=%.6f, wall=%.6f s\n', ...
    telemetryAfter.timing.initialization_time, ...
    telemetryAfter.timing.optimization_loop_time, ...
    telemetryAfter.timing.postprocessing_time, wallAfter);
fprintf('  PASS: trajectory, iterations, objective history, frequencies, and topology are identical.\n');

validation = struct();
validation.case_description = 'Proposed, 160x20, diagnostics enabled versus benchmark diagnostics disabled';
validation.iterations_before = nIterBefore;
validation.iterations_after = nIterAfter;
validation.objective_before = telemetryBefore.objective_final;
validation.objective_after = telemetryAfter.objective_final;
validation.objective_history_max_abs_difference = objectiveMaxAbsDiff;
validation.frequencies_before_rad_s = omegaBefore(:)';
validation.frequencies_after_rad_s = omegaAfter(:)';
validation.frequency_max_abs_difference = frequencyMaxAbsDiff;
validation.topology_checksum_before = numeric_fingerprint(xBefore);
validation.topology_checksum_after = numeric_fingerprint(xAfter);
validation.topology_max_abs_difference = topologyMaxAbsDiff;
validation.runtime_before = struct( ...
    'legacy_reconstructed_time_s', tIterBefore*nIterBefore, ...
    'total_wall_time_s', wallBefore);
validation.runtime_after = struct( ...
    'initialization_time_s', telemetryAfter.timing.initialization_time, ...
    'optimization_loop_time_s', telemetryAfter.timing.optimization_loop_time, ...
    'postprocessing_time_s', telemetryAfter.timing.postprocessing_time, ...
    'total_wall_time_s', wallAfter, ...
    'legacy_reconstructed_time_s', tIterAfter*nIterAfter);
validation.pass = true;

validationPath = fullfile(thisDir, 'instrumentation_validation.json');
fid = fopen(validationPath, 'w');
assert(fid >= 0, 'validate_instrumentation_invariance:OpenFailed', ...
    'Cannot open %s for writing.', validationPath);
fprintf(fid, '%s\n', jsonencode(validation));
fclose(fid);
fprintf('  Validation record saved to: %s\n', validationPath);

function value = numeric_fingerprint(x)
    x = double(x(:));
    weights = (1:numel(x))';
    value = sprintf('n=%d;sum=%.17g;weighted=%.17g;l2=%.17g', ...
        numel(x), sum(x), sum(weights.*x), norm(x));
end
