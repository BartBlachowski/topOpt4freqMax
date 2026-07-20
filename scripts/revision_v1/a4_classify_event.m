function [classes, details] = a4_classify_event(event, candidates)
%A4_CLASSIFY_EVENT  Exact Phase-2 event taxonomy (§7.3).
% E-1 replaces E-0 when the selected index exceeds m0. E-4 is orthogonal and
% may co-occur with E-2. E-3 necessarily adds E-5.

c = a4_phase2_constants();
classes = {};
details = struct('best_mac_index', 0, 'best_mac_support_kinetic_fraction', NaN, ...
    'n_solid_components', event.n_solid_components, 'reason', '');

switch event.search_outcome
    case 'SOLVER_FAILURE'
        classes = {'E-5'};
        details.reason = event.failure_message;
        return;
    case 'REFERENCE_UNAVAILABLE'
        physical = false(size(candidates));
        for k = 1:numel(candidates)
            physical(k) = candidates(k).cond_kinetic_pass && ...
                candidates(k).cond_supports_pass && candidates(k).cond_strain_pass;
        end
        if any(physical)
            classes = {'E-2b'};
            details.reason = 'at least one physical candidate exists; none passes MAC continuity';
        else
            classes = {'E-2a'};
            details.reason = 'no candidate passes the three physical-mode conditions';
        end
    case 'SELECTED'
        if event.selected_index < 1 || event.selected_index > numel(candidates)
            classes = {'E-5'};
            details.reason = 'SELECTED outcome has no valid selected candidate';
            return;
        end
        selected = candidates(event.selected_index);
        if event.reference_changed && ~selected.admissible
            classes = {'E-3', 'E-5'};
            details.reason = 'an inadmissible candidate was adopted';
            return;
        elseif event.selected_index > c.m0
            classes = {'E-1'};
            details.reason = sprintf('selected index %d exceeds old window %d', ...
                event.selected_index, c.m0);
        elseif strcmp(event.stability_flag, 'confirmed')
            classes = {'E-0'};
            details.reason = 'confirmed admissible selection within old window';
        else
            % Under nested spectra an unconfirmed M_max selection must have
            % entered above the preceding rung. Treat violation as machinery,
            % not as a new scientific class (§7 taxonomy is exhaustive).
            classes = {'E-5'};
            details.reason = 'unconfirmed selection at or below m0 violates ladder invariants';
            return;
        end
    otherwise
        classes = {'E-5'};
        details.reason = sprintf('unknown search outcome "%s"', event.search_outcome);
        return;
end

% E-4 is an orthogonal mechanism and may accompany E-2 (§7.3).
if event.n_solid_components >= 2 && ~isempty(candidates)
    macs = [candidates.mac_prev];
    [~, bestIdx] = max(macs); % lower-index deterministic tie from MATLAB max
    details.best_mac_index = bestIdx;
    details.best_mac_support_kinetic_fraction = ...
        candidates(bestIdx).support_kinetic_fraction;
    if isfinite(details.best_mac_support_kinetic_fraction) && ...
            details.best_mac_support_kinetic_fraction < c.tau_kin
        classes{end+1} = 'E-4'; %#ok<AGROW>
    end
end
end
