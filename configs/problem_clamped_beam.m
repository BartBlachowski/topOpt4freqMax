function prob = problem_clamped_beam(nelx, nely, volfrac)
%PROBLEM_CLAMPED_BEAM  Clamped beam of Section 4.2 (Du & Olhoff 2007, Fig. 2c).
%   8 m x 1 m, both end faces clamped, E = 1e7 Pa, nu = 0.3, rho = 1 kg/m^3,
%   void floor Emin = 1e-6 E0, rhoMin = 1e-6.  Defaults: 400 x 50, V_f = 0.5.
if nargin < 1 || isempty(nelx), nelx = 400; end
if nargin < 2 || isempty(nely), nely = nelx/8; end
if nargin < 3 || isempty(volfrac), volfrac = 0.5; end
prob = struct('name', 'clamped_beam', 'L', 8, 'H', 1, 'nelx', nelx, 'nely', nely, ...
    'E', 1e7, 'nu', 0.3, 'rho', 1, 'volfrac', volfrac, 'support', 'clamped', ...
    'passive', zeros(0,1), 'Emin', 1e-6*1e7, 'rhoMin', 1e-6);
end
