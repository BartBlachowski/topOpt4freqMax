function [cfg, meta] = repro2007_config(name, overrides)
%REPRO2007_CONFIG  Named, documented configurations of the clean-room reproduction.
%
%   cfg = REPRO2007_CONFIG() returns the default reproduction configuration,
%   'fig3a_best'.
%
%   cfg = REPRO2007_CONFIG(name) returns the named configuration.
%   cfg = REPRO2007_CONFIG(name, overrides) applies the fields of the struct
%   OVERRIDES on top of it.  Unknown field names are rejected, so a typo in a
%   sweep script cannot silently run the default instead.
%
%   [cfg, meta] = REPRO2007_CONFIG(...) also returns provenance for the named
%   configuration: which frozen artifact under baseline/ it reproduces, and the
%   frequencies that artifact holds.
%
%   Every configuration below is anchored to an artifact that the clean-room
%   study actually produced.  None of these numbers is a fresh choice made
%   during migration -- see PROVENANCE.md and NOTES.md.
%
%   AVAILABLE CONFIGURATIONS
%   ------------------------
%   'fig3a_best'   DEFAULT.  The best single reproduction of the paper's Fig. 3a
%                  optimum.  240x30, r_min = 1.3 elements, move = 0.005,
%                  1600 outer iterations, Eq. (22) LP route.
%                  Reproduces omega = 170.4709 / 170.8659 / 285.1939 against
%                  the paper's 174.7 / 174.7 / 284.9 -- bimodal, 0.23% gap,
%                  omega_3 to +0.1%.        Frozen: baseline/lp240_rmin1.3.mat
%
%   'fig4_history' The trajectory that reproduces the paper's Fig. 4 iteration
%                  history.  Same model, move = 0.02 (the move limit sets the
%                  pace of that figure and the paper never states it -- see
%                  NOTES.md 8c), 400 outer iterations.
%                  omega_1/omega_2 coalesce at iteration 26 against the paper's
%                  ~20.                    Frozen: baseline/FIG4_definitive.mat
%
%   'rmin1p8'      The r_min = 1.8 element point of the filter-radius sweep,
%                  specified in PHYSICAL units (0.06) as it was when run.
%                  Retained because it is the artifact the source repository
%                  named FINAL_lp_240x30, and because it is the "large filter"
%                  end of the transition documented in NOTES.md 6.
%                  Frozen: baseline/FINAL_lp_240x30.mat
%
%   'paper_mma'    The paper-literal route: MMA inner loop with the full
%                  Eq. (25d) coupling, i.e. defaultCfg() exactly as the
%                  clean-room study shipped it.  This is the LABELLED BASELINE
%                  of the reproduction, not its successful configuration: it
%                  does not converge once N >= 2 (NOTES.md 7).  Provided so the
%                  documented alternative stays executable.   No frozen artifact.
%
%   'migration_smoke'
%                  NOT A REPRODUCTION CONFIGURATION.  A short 160x20 / 12-outer
%                  trajectory used only to check that migration did not perturb
%                  the numerics.  Never cite it as a result.
%
%   See also RUN_REPRO2007, DEFAULTCFG, REPRO2007_REGRESSION.

if nargin < 1 || isempty(name)
    name = 'fig3a_best';
end
name = char(string(name));

if exist('defaultCfg', 'file') ~= 2
    error('repro2007_config:PathNotInstalled', ...
        ['defaultCfg is not on the MATLAB path.  Call guard = repro2007_paths() ' ...
         'first, or use run_repro2007, which does it for you.']);
end

% Start from the clean-room baseline, imported verbatim.  Every named
% configuration below is expressed as a DIFFERENCE from it, so that reading
% this file tells you exactly what each reproduction changed.
cfg = defaultCfg();
meta = struct('name', name, 'baseline_artifact', '', 'omega_expected', [], ...
    'source_reference', '', 'is_reproduction', true);

switch lower(name)
    case 'fig3a_best'
        cfg.nelx = 240;   cfg.nely = 30;
        cfg.rminEl = 1.3; cfg.rminPhys = [];
        cfg.move = 0.005;
        cfg.tolMult = 0.05;
        cfg.maxOuter = 1600;
        cfg.innerSolver = 'lp';
        cfg.offDiag = false;
        cfg.filterMode = 'diag';
        meta.baseline_artifact = 'baseline/lp240_rmin1.3.mat';
        meta.omega_expected = [170.4709086; 170.8658865; 285.1939392];
        meta.source_reference = 'NOTES.md 6 and 9; PROVENANCE.md';

    case 'fig4_history'
        cfg.nelx = 240;   cfg.nely = 30;
        cfg.rminEl = 1.3; cfg.rminPhys = [];
        cfg.move = 0.02;
        cfg.tolMult = 0.05;
        cfg.maxOuter = 400;
        cfg.innerSolver = 'lp';
        cfg.offDiag = false;
        cfg.filterMode = 'diag';
        meta.baseline_artifact = 'baseline/FIG4_definitive.mat';
        meta.omega_expected = [170.744886092; 175.06561593; 301.871711545];
        meta.source_reference = 'NOTES.md 8c';

    case 'rmin1p8'
        cfg.nelx = 240;   cfg.nely = 30;
        cfg.rminEl = 1.8; cfg.rminPhys = 0.06;   % physical spec, as run
        cfg.move = 0.005;
        cfg.tolMult = 0.05;
        cfg.maxOuter = 1600;
        cfg.innerSolver = 'lp';
        cfg.offDiag = false;
        cfg.filterMode = 'diag';
        meta.baseline_artifact = 'baseline/FINAL_lp_240x30.mat';
        meta.omega_expected = [167.868341697; 172.7672778; 257.317406555];
        meta.source_reference = 'NOTES.md 6';

    case 'paper_mma'
        % defaultCfg() unchanged: innerSolver 'mma', offDiag true, rminEl 3.0,
        % move 0.05, tolMult 0.02, 160x20, maxOuter 200.
        meta.source_reference = 'algo/defaultCfg.m (imported verbatim); NOTES.md 7';

    case 'migration_smoke'
        cfg.nelx = 160;   cfg.nely = 20;
        cfg.rminEl = 1.2; cfg.rminPhys = [];
        cfg.move = 0.005;
        cfg.tolMult = 0.05;
        cfg.maxOuter = 12;
        cfg.innerSolver = 'lp';
        cfg.offDiag = false;
        cfg.filterMode = 'diag';
        cfg.verbose = false;
        meta.is_reproduction = false;
        meta.source_reference = 'migration check only -- not a published result';

    otherwise
        error('repro2007_config:UnknownConfiguration', ...
            ['Unknown configuration "%s".  Available: fig3a_best, ' ...
             'fig4_history, rmin1p8, paper_mma, migration_smoke.'], name);
end

cfg.name = name;

if nargin >= 2 && ~isempty(overrides)
    if ~isstruct(overrides) || numel(overrides) ~= 1
        error('repro2007_config:InvalidOverrides', ...
            'overrides must be a scalar struct.');
    end
    fn = fieldnames(overrides);
    known = fieldnames(cfg);
    unknown = setdiff(fn, known);
    if ~isempty(unknown)
        error('repro2007_config:UnknownOverrideField', ...
            ['Unknown configuration field(s): %s.  Refusing to run rather ' ...
             'than silently ignoring them.'], strjoin(unknown', ', '));
    end
    for i = 1:numel(fn)
        cfg.(fn{i}) = overrides.(fn{i});
    end
    % An override invalidates the frozen-artifact claim.
    if ~isempty(meta.baseline_artifact)
        meta.baseline_artifact = '';
        meta.omega_expected = [];
        meta.is_reproduction = false;
    end
end
end
