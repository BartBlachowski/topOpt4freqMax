function cfg = repro2007_direct_cfg(data, verbose)
%REPRO2007_DIRECT_CFG  The clean-room configuration a benchmark task profile
%   asks for, built WITHOUT the benchmark dispatcher.
%
%   cfg = REPRO2007_DIRECT_CFG(data)
%   cfg = REPRO2007_DIRECT_CFG(data, verbose)
%
%   DATA is a task profile as PERFORMANCE_BENCHMARK_PROFILE returns it.  The
%   result is the struct to hand straight to OLHOFFOPT.
%
%   REQUIRES an installed reproduction path -- the caller must be holding a
%   REPRO2007_PATHS guard, because REPRO2007_CONFIG needs DEFAULTCFG.
%
%   INDEPENDENCE IS THE POINT
%   -------------------------
%   This is a SECOND, deliberately separate reading of the task profile.  It
%   does not call, share or consult RUN_TOPOPT_FROM_JSON's mapping table.  If
%   the two paths derived their configuration from one piece of code, the
%   equivalence check could not detect a configuration-mapping defect -- and a
%   configuration-mapping defect is exactly what it was built to detect
%   (DIAGNOSTIC_REPRO2007_BENCHMARK.md).
%
%   WHAT IT SETS, AND WHAT IT REFUSES TO SET
%   ----------------------------------------
%   Only the quantities the benchmark profile actually states: geometry, mesh,
%   material, volume fraction, penalization, filter radius, and the four
%   method-scoped settings under optimization.repro2007 that cannot be
%   inherited from the shared block.  Everything else -- target mode n, Nmax,
%   tolMult, offDiag, innerSolver, filterMode, massInterp, rho0, support,
%   axial, elemType, massType, maxInner, tolInner, minInner, eigensolver,
%   threads -- keeps its documented `fig3a_best` value.  If the benchmark path
%   changes one of those, the config hashes diverge and the gate fails, which
%   is the intended behaviour.
%
%   See also PERFORMANCE_BENCHMARK_PROFILE, REPRO2007_CONFIG,
%            REPRO2007_NORMALIZED_CONFIG, VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE.

if nargin < 2 || isempty(verbose)
    verbose = false;
end

cfg = repro2007_config('fig3a_best');

% --- geometry, mesh, material: read straight off the task profile ---------
cfg.nelx    = data.domain.mesh.nelx;
cfg.nely    = data.domain.mesh.nely;
cfg.a       = data.domain.size.length;
cfg.b       = data.domain.size.height;
cfg.t       = data.domain.thickness;
cfg.E       = data.material.E;
cfg.nu      = data.material.nu;
cfg.rhom    = data.material.rho;
cfg.volfrac = data.optimization.volume_fraction;
cfg.p       = data.optimization.penalization;

% --- filter radius, in whichever units the profile states them -----------
units = lower(char(string(data.optimization.filter.radius_units)));
switch units
    case 'element'
        cfg.rminEl   = data.optimization.filter.radius;
        cfg.rminPhys = [];
    case 'physical'
        cfg.rminEl   = [];
        cfg.rminPhys = data.optimization.filter.radius;
    otherwise
        error('repro2007_direct_cfg:BadRadiusUnits', ...
            'optimization.filter.radius_units must be "element" or "physical".');
end

% --- the method-scoped block: the settings that cannot be inherited ------
r = data.optimization.repro2007;
cfg.move     = r.move;
cfg.maxOuter = r.max_outer;
cfg.rhomin   = r.rho_min;
cfg.tolOuter = r.tol_outer;
switch upper(char(string(r.support_type)))
    case 'SS', cfg.bc = 'a';
    case 'CS', cfg.bc = 'b';
    case 'CC', cfg.bc = 'c';
    otherwise
        error('repro2007_direct_cfg:BadSupport', ...
            'optimization.repro2007.support_type must be SS, CS or CC.');
end

% --- reporting only ------------------------------------------------------
% Left at the named configuration's value unless the caller asks for quiet,
% mirroring the dispatcher, which touches verbose only when
% postprocessing.visualize_live is true.  Excluded from the config hash.
if ~verbose
    cfg.verbose = false;
end
cfg.name = 'fig3a_best';
end
