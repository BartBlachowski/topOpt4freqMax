function [rho_e, drho_dx, m, dm_dx] = our_mass_interpolation(x, rho0, rho_min, mode, pmass)
%OUR_MASS_INTERPOLATION Mass interpolation used by ourApproach.
%
% The default power law preserves the previous implementation:
%   rho_e = rho_min + x^pmass * (rho0 - rho_min).
%
% The du2007_c1 option ports Du & Olhoff Eq. 4b from
% analysis/OlhoffApproachExact/Matlab/mass_interp.m, then applies the same
% rho_min/rho0 scaling used by ourApproach.

if nargin < 4 || isempty(mode)
    mode = 'power';
end
if nargin < 5 || isempty(pmass)
    pmass = 1.0;
end

x = double(x);
modeKey = lower(strtrim(char(string(mode))));
pmass = double(pmass);
if ~isfinite(pmass) || pmass <= 0
    error('our_mass_interpolation:InvalidPmass', ...
        'pmass must be a positive finite scalar.');
end

switch modeKey
    case {'linear'}
        m = x;
        dm_dx = ones(size(x));

    case {'power', 'simp_power', 'pmass'}
        m = x .^ pmass;
        dm_dx = pmass .* (x .^ (pmass - 1));

    case {'du2007_c1', 'du_olhoff_c1', 'eq4b'}
        c1 =  6e5;
        c2 = -5e6;
        hi = x > 0.1;
        lo = ~hi;
        m = zeros(size(x));
        dm_dx = zeros(size(x));
        m(hi) = x(hi);
        dm_dx(hi) = 1;
        m(lo) = c1 .* x(lo).^6 + c2 .* x(lo).^7;
        dm_dx(lo) = 6*c1 .* x(lo).^5 + 7*c2 .* x(lo).^6;

    otherwise
        error('our_mass_interpolation:InvalidMode', ...
            'Unknown mass interpolation mode "%s".', modeKey);
end

rho_e = rho_min + m .* (rho0 - rho_min);
drho_dx = dm_dx .* (rho0 - rho_min);
end
