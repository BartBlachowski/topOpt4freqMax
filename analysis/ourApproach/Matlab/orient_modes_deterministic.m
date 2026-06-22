function modes = orient_modes_deterministic(modes)
%ORIENT_MODES_DETERMINISTIC Make each largest-magnitude mode DOF nonnegative.
if ~isreal(modes) || ~isnumeric(modes) || isempty(modes) || ...
        ~ismatrix(modes) || any(~isfinite(modes(:)))
    error('modal_utils:InvalidModes', 'Modes must be a finite, non-empty real matrix.');
end
for k = 1:size(modes,2)
    [~, phaseIdx] = max(abs(modes(:,k)));
    if modes(phaseIdx,k) < 0
        modes(:,k) = -modes(:,k);
    end
end
end
