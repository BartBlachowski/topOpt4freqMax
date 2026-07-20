function [event, candidates] = a4_build_event_telemetry(search, meta)
%A4_BUILD_EVENT_TELEMETRY  Materialize Phase-2 §6 event/candidate records.
% All candidates come from search.m_final. Unavailable measurements are NaN
% (MAT/CSV) and therefore JSON null; this routine never fabricates defaults.

event = localBlankEvent();
event.arm_N = meta.arm_N;
event.iteration = meta.iteration;
event.event_kind = meta.event_kind;
event.event_id = meta.event_id;
event.window_rungs_solved = search.window_rungs_solved;
event.m_final = search.m_final;
event.search_outcome = search.search_outcome;
event.stability_flag = search.stability_flag;
event.stability_mac = search.stability_mac;
event.n_candidates = search.n_candidates;
event.n_admissible = search.n_admissible;
event.selected_index = search.selected_index;
event.omega_min = localFirst(search.omegas);
event.omega1_omega2_gap = localGap(search.omegas);
event.n_solid_components = localScreenField(search, 'nComponents', 0);
event.reference_changed = logical(localMeta(meta, 'reference_changed', false));
event.deferred = logical(localMeta(meta, 'deferred', false));
event.eigensolve_count_at_event = search.eigensolve_count;
event.wall_clock_s = localMeta(meta, 'wall_clock_s', NaN);
event.max_design_change = localMeta(meta, 'max_design_change', NaN);
event.feasibility_relative = localMeta(meta, 'feasibility_relative', NaN);
event.surrogate_objective = localMeta(meta, 'surrogate_objective', NaN);
event.omitted_term_ratio = localMeta(meta, 'omitted_term_ratio', NaN);
event.min_x_e = localMeta(meta, 'min_x_e', NaN);
event.failure_identifier = search.failure_identifier;
event.failure_message = search.failure_message;
event.tie_flag = search.tie_flag;

if search.selected_index > 0 && search.selected_index <= numel(search.candidates)
    selected = search.candidates(search.selected_index);
    event.selected_omega = selected.omega;
    event.selected_mac_prev = selected.mac_prev;
    event.selected_mac_phi0 = selected.mac_phi0;
    event.low_density_kinetic_fraction = selected.low_density_kinetic_fraction;
    event.omega_tracked_minus_min = selected.omega - event.omega_min;
end

candidates = repmat(localBlankCandidate(), numel(search.candidates), 1);
for k = 1:numel(search.candidates)
    src = search.candidates(k);
    dst = localBlankCandidate();
    dst.arm_N = meta.arm_N;
    dst.iteration = meta.iteration;
    dst.event_kind = meta.event_kind;
    dst.event_id = meta.event_id;
    dst.window_m_final = search.m_final;
    dst.mode_index = src.index;
    dst.omega = src.omega;
    dst.mac_prev = src.mac_prev;
    dst.mac_phi0 = src.mac_phi0;
    dst.mac_solid = src.mac_solid;
    dst.support_kinetic_fraction = src.largest_support_component_kinetic_fraction;
    dst.low_density_strain_fraction = src.low_density_strain_fraction;
    dst.low_density_kinetic_fraction = src.low_density_kinetic_fraction;
    dst.support_connectivity = logical(src.dominant_component_touches_both_supports);
    dst.cond_kinetic_pass = logical(src.cond_kinetic_pass);
    dst.cond_supports_pass = logical(src.cond_supports_pass);
    dst.cond_strain_pass = logical(src.cond_strain_pass);
    dst.cond_mac_pass = logical(src.cond_mac_pass);
    dst.rejection_reason = src.reject_reason;
    dst.admissible = logical(src.admissible);
    dst.selected = logical(src.selected);
    dst.tie_flag = logical(src.tie_flag);
    dst.eigensolver_status = src.eigensolver_status;
    candidates(k) = dst;
end
end

function e = localBlankEvent()
e = struct('arm_N', '', 'iteration', 0, 'event_kind', '', 'event_id', 0, ...
    'window_rungs_solved', [], 'm_final', 0, 'search_outcome', '', ...
    'stability_flag', 'n/a', 'stability_mac', NaN, 'n_candidates', 0, ...
    'n_admissible', 0, 'selected_index', 0, 'selected_omega', NaN, ...
    'selected_mac_prev', NaN, 'selected_mac_phi0', NaN, 'omega_min', NaN, ...
    'omega_tracked_minus_min', NaN, 'omega1_omega2_gap', NaN, ...
    'n_solid_components', 0, 'low_density_kinetic_fraction', NaN, ...
    'reference_changed', false, 'deferred', false, ...
    'eigensolve_count_at_event', 0, 'wall_clock_s', NaN, ...
    'max_design_change', NaN, 'feasibility_relative', NaN, ...
    'surrogate_objective', NaN, 'omitted_term_ratio', NaN, 'min_x_e', NaN, ...
    'tie_flag', false, 'event_classes', {{}}, ...
    'failure_identifier', '', 'failure_message', '');
end

function r = localBlankCandidate()
% Field order is the exact CSV order required by Phase-2 specification §6.1.
r = struct('arm_N', '', 'iteration', 0, 'event_kind', '', 'event_id', 0, ...
    'window_m_final', 0, 'mode_index', 0, 'omega', NaN, 'mac_prev', NaN, ...
    'mac_phi0', NaN, 'mac_solid', NaN, 'support_kinetic_fraction', NaN, ...
    'low_density_strain_fraction', NaN, 'low_density_kinetic_fraction', NaN, ...
    'support_connectivity', false, 'cond_kinetic_pass', false, ...
    'cond_supports_pass', false, 'cond_strain_pass', false, ...
    'cond_mac_pass', false, 'rejection_reason', '', 'admissible', false, ...
    'selected', false, 'tie_flag', false, 'eigensolver_status', '');
end

function v = localMeta(s, name, default)
v = default;
if isstruct(s) && isfield(s, name) && ~isempty(s.(name)), v = s.(name); end
end

function v = localScreenField(search, name, default)
v = default;
if isfield(search, 'screen') && isstruct(search.screen) && ...
        isfield(search.screen, name) && ~isempty(search.screen.(name))
    v = search.screen.(name);
end
end

function v = localFirst(x)
v = NaN;
if ~isempty(x), v = x(1); end
end

function v = localGap(x)
v = NaN;
if numel(x) >= 2, v = x(2) - x(1); end
end
