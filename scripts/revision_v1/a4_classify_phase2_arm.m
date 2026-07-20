function arm = a4_classify_phase2_arm(arm)
%A4_CLASSIFY_PHASE2_ARM  Mechanical Phase-2 arm acceptance (§8.1--§8.4).

c = a4_phase2_constants();
events = arm.screening_events;
if isempty(events)
    classes = {};
else
    classes = localAllClasses(events);
end
failures = {};
if any(strcmp(classes, 'E-5'))
    failures{end+1} = 'E-5 event recorded'; %#ok<AGROW>
end
terminal = arm.success && arm.iterations > 0 && ...
    (arm.iterations >= arm.cap || ...
     (isfinite(arm.final_design_change) && arm.final_design_change < arm.tol));
if ~terminal, failures{end+1} = 'arm did not reach convergence or iteration cap'; end %#ok<AGROW>

if isempty(events)
    operational = [];
else
    operational = events(ismember({events.event_kind}, {'operational','both'}));
end
if numel(operational) ~= arm.n_refresh_scheduled
    failures{end+1} = 'scheduled refresh telemetry count mismatch'; %#ok<AGROW>
end
expectedGrid = unique([c.diagnostic_grid(c.diagnostic_grid <= arm.iterations), arm.iterations]);
if isempty(events)
    diagnosticIterations = [];
else
    diagnostic = events(ismember({events.event_kind}, {'diagnostic','both'}));
    diagnosticIterations = sort(unique([diagnostic.iteration]));
end
if ~isequal(diagnosticIterations, expectedGrid)
    failures{end+1} = 'diagnostic grid incomplete'; %#ok<AGROW>
end
if numel(arm.iteration_histories) ~= arm.iterations
    failures{end+1} = 'per-iteration histories incomplete'; %#ok<AGROW>
end
endpointFields = {'omega1_tracked','omega1_min','omega1_thresholded', ...
    'omega1_omega2_gap','mode_index_jstar','mac_to_phi0','grayness','feasibility'};
endpointFields{end+1} = 'mac_thresholded_to_phi0';
for k = 1:numel(endpointFields)
    if ~isfield(arm, endpointFields{k}) || ~isfinite(arm.(endpointFields{k}))
        failures{end+1} = sprintf('endpoint missing/nonfinite: %s', endpointFields{k}); %#ok<AGROW>
    end
end

arm.degenerate = arm.n_refresh_scheduled > 0 && ...
    arm.n_deferred == arm.n_refresh_scheduled;
arm.warnings = {};
if arm.n_deferred >= 1 && arm.n_deferred < arm.n_refresh_scheduled
    arm.warnings{end+1} = 'W-1';
end
if ~isempty(events)
    if any([events.m_final] == c.M_max), arm.warnings{end+1} = 'W-2'; end
    if any(strcmp({events.stability_flag}, 'unconfirmed')), arm.warnings{end+1} = 'W-3'; end
    deep = arrayfun(@(e) any(strcmp(e.event_classes, 'E-1')) && ...
        e.selected_index > c.M_max/2, events);
    if any(deep), arm.warnings{end+1} = 'W-4'; end
    hasE4 = arrayfun(@(e) any(strcmp(e.event_classes, 'E-4')), events);
    if any(hasE4), arm.warnings{end+1} = 'W-5'; end
end
arm.warnings = unique(arm.warnings, 'stable');
arm.implementation_failures = failures;

if ~isempty(failures)
    arm.phase2_status = 'REJECTED';
elseif arm.degenerate
    arm.phase2_status = 'UNAVAILABLE';
elseif isempty(arm.warnings) && arm.n_deferred == 0
    arm.phase2_status = 'ACCEPTED';
else
    arm.phase2_status = 'ACCEPTED_WITH_WARNING';
end
end

function classes = localAllClasses(events)
classes = {};
for i = 1:numel(events), classes = [classes, events(i).event_classes]; end %#ok<AGROW>
end
