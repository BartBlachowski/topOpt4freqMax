function modes = mass_normalize_modes(modes, M)
%MASS_NORMALIZE_MODES Scale each mode so phi' * M * phi equals one.
if ~isnumeric(modes) || isempty(modes) || ~ismatrix(modes) || any(~isfinite(modes(:)))
    error('modal_utils:InvalidModes', 'Modes must be a finite, non-empty numeric matrix.');
end
if ~ismatrix(M) || size(M,1) ~= size(M,2) || size(M,1) ~= size(modes,1)
    error('modal_utils:InvalidMassMatrix', 'Mass matrix dimensions do not match the mode DOFs.');
end
for k = 1:size(modes,2)
    modalMass = real(modes(:,k)' * (M * modes(:,k)));
    if ~isfinite(modalMass) || modalMass <= 0
        error('modal_utils:InvalidModalMass', ...
            'Mode %d has invalid modal mass %.16g.', k, modalMass);
    end
    modes(:,k) = modes(:,k) / sqrt(modalMass);
end
end
