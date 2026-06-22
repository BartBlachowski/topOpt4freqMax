function mac = squared_mass_weighted_mac(phi, psi, M)
%SQUARED_MASS_WEIGHTED_MAC Compute pairwise squared mass-weighted MAC.
if ~isnumeric(phi) || ~isnumeric(psi) || isempty(phi) || isempty(psi) || ...
        ~ismatrix(phi) || ~ismatrix(psi) || any(~isfinite(phi(:))) || any(~isfinite(psi(:)))
    error('modal_utils:InvalidModes', 'Mode sets must be finite, non-empty numeric matrices.');
end
if size(phi,1) ~= size(psi,1) || ~ismatrix(M) || ...
        size(M,1) ~= size(M,2) || size(M,1) ~= size(phi,1)
    error('modal_utils:DimensionMismatch', 'Mass matrix and mode-set dimensions do not agree.');
end
massPhi = real(sum(phi .* (M * phi), 1));
massPsi = real(sum(psi .* (M * psi), 1));
if any(~isfinite(massPhi)) || any(massPhi <= 0) || ...
        any(~isfinite(massPsi)) || any(massPsi <= 0)
    error('modal_utils:InvalidModalMass', 'Every mode must have positive finite modal mass.');
end
cross = real(phi' * (M * psi));
mac = cross.^2 ./ (massPhi' * massPsi);
if any(~isfinite(mac(:)))
    error('modal_utils:NonfiniteMAC', 'MAC calculation produced non-finite values.');
end
mac = min(max(mac, 0), 1);
end
