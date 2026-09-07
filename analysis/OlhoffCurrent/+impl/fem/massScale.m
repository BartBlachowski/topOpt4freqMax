function [g, dg] = massScale(rho, massInterp)
%MASSSCALE  Mass interpolation factor g(rho) and its derivative.
%
%   [g,dg] = MASSSCALE(rho, massInterp)
%
%   massInterp may be
%     * a canonical cfg.material.mass struct  (model/q/lowDensityExponent/cutoff)
%     * a legacy model name: 'lin' | '4' | '4a' | '4b'
%
%   This function is now a THIN DISPATCHER.  The piecewise formulas, the printed
%   coefficients and the provenance comments live in exactly one place,
%   olh.material.massInterpolation, so value and derivative cannot drift apart
%   and no caller needs to know the formula.
%
%   Legacy name -> printed equation:
%       'lin' -> eq. (2) with q = 1        '4'  -> eq. (4)
%       '4a'  -> eq. (4a), C0              '4b' -> eq. (4b), C1

if isstruct(massInterp)
    mass = massInterp;
else
    switch lower(char(massInterp))
        case 'lin', model = 'eq2';
        case '4',   model = 'eq4';
        case '4a',  model = 'eq4a';
        case '4b',  model = 'eq4b';
        otherwise
            error('massScale:model','unknown mass interpolation %s',char(massInterp));
    end
    % The printed values, which are the only ones the legacy flat configuration
    % could ever have expressed: q = 1 (eq. 2), r = 6 and a cut-off at 0.1.
    mass = struct('model',model,'q',1,'lowDensityExponent',6,'cutoff',0.1);
end

[g, dg] = olh.material.massInterpolation(rho, mass);
end
