function prob = problem_ss_beam(nelx, nely, volfrac)
%PROBLEM_SS_BEAM  Simply supported beam of Section 4.1 (Du & Olhoff 2007, Fig. 2a).
%   8 m x 1 m, unit thickness, E = 1e7 Pa, nu = 0.3, rho = 1 kg/m^3, pins (ux, uy)
%   at mid-height of both end faces.  Defaults: 400 x 50 elements, V_f = 0.5.
%   Emin/rhoMin: material floor of the proposed and Yuksel-Yilmaz methods (1e-9).
if nargin < 1 || isempty(nelx), nelx = 400; end
if nargin < 2 || isempty(nely), nely = nelx/8; end
if nargin < 3 || isempty(volfrac), volfrac = 0.5; end
prob = struct('name', 'ss_beam', 'L', 8, 'H', 1, 'nelx', nelx, 'nely', nely, ...
    'E', 1e7, 'nu', 0.3, 'rho', 1, 'volfrac', volfrac, 'support', 'simplySupported', ...
    'passive', zeros(0,1), 'Emin', 1e-9*1e7, 'rhoMin', 1e-9);
end
