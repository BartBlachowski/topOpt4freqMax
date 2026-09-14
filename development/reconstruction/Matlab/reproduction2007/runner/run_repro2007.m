function [x, omega, tIter, nIter, info] = run_repro2007(runCfg)
%RUN_REPRO2007  Standardized runner for the Du-Olhoff 2007 clean-room reproduction.
%
%   [x, omega, tIter, nIter, info] = RUN_REPRO2007(runCfg)
%
%   Wraps the clean-room implementation (Eq. 22 LP route) in the same calling
%   convention that tools/Matlab/run_topopt_from_json.m already uses for the
%   Yuksel and Proposed methods:
%
%     x       final density vector, nelx*nely x 1, column-major (e = (i-1)*nely+j)
%     omega   first three circular frequencies of the final design [rad/s]
%     tIter   mean wall time per outer iteration [s]
%     nIter   number of outer iterations executed
%     info    struct described under OUTPUT INFO below
%
%   THIS FUNCTION DOES NOT IMPLEMENT ANY PART OF THE ALGORITHM.  It installs an
%   isolated path, translates repository-standard option names onto the
%   configuration struct of the imported implementation, calls OLHOFFOPT
%   unchanged, and reshapes what OLHOFFOPT recorded into the repository's
%   history schema.  Every number it reports is either produced by OLHOFFOPT or
%   derived from OLHOFFOPT's own record.
%
%   INPUT  runCfg  (struct, all fields optional)
%   ---------------------------------------------
%   Configuration selection
%     config          name passed to REPRO2007_CONFIG (default 'fig3a_best')
%
%   Parameters required for the parametric study (WP3).  Each maps onto exactly
%   one field of the imported configuration; anything left unset keeps the
%   named configuration's documented value.
%     nelx, nely      mesh dimensions
%     volfrac         volume fraction (paper: alpha = 0.5)
%     target_mode     n, the eigenfrequency being maximized (paper Fig. 3a: 1)
%     rmin_elem       filter radius in ELEMENT units       -> cfg.rminEl
%     rmin_phys       filter radius in PHYSICAL units      -> cfg.rminPhys
%                     (rminPhys overrides rminEl inside OLHOFFOPT when set)
%     move            move limit on drho
%     tol_mult        multiplicity tolerance (relative)
%     max_outer       maximum outer iterations
%     tol_outer       outer convergence tolerance on max|drho|
%     mass_interp     '4' | '4a' | '4b' | 'lin'            -> cfg.massInterp
%     filter_mode     'diag' | 'all' | 'none'              -> cfg.filterMode
%                     (which generalized gradients f_sk are filtered)
%
%   Route and model options
%     inner_solver    'lp' (Eq. 22 route) | 'mma'          -> cfg.innerSolver
%     off_diag        true = full Eq. (25d) coupling       -> cfg.offDiag
%     n_modes_max     Nmax, the multiplicity budget        -> cfg.Nmax
%     penal           SIMP exponent p
%     rho_min         void density lower bound
%     rho0            uniform initial density
%     E0, nu, rho_m, L, H, thickness
%     support_type    'SS' -> bc 'a', 'CS' -> 'b', 'CC' -> 'c'
%     support         'mid' | 'corner' | 'face'
%     axial           'one' | 'both'
%     elem_type       'Q4' | 'Q6'
%     mass_type       'consistent' | 'lumped'
%     eig_solver      'eigs' | 'dense'
%     threads         BLAS threads (the reproduction pins this to 1)
%
%   Reporting
%     verbose         echo the per-iteration table (default true)
%     approach_name   label used in console output
%     record_history  build info.history (default true)
%     save_results    save info to results_dir (default false)
%     results_dir     defaults to <root>/results
%     run_label       basename for the saved file
%
%   OUTPUT INFO
%   -----------
%     .native            the verbatim struct returned by OLHOFFOPT, untouched
%     .cfg               the configuration actually executed
%     .config_meta       provenance of the named configuration
%     .history           per-outer-iteration table, schema below (WP4)
%     .timing            initialization_time / loop_time / postprocessing_time
%     .stopping          stop classification and final acceptance quantities.
%                        `status` is the precedence-ordered verdict
%                        SOLVER_FAILURE > CONVERGED > CAP_HIT > RUNNING;
%                        `stop_reason` keeps the repository vocabulary; and the
%                        raw native quantities the verdict came from
%                        (native_stop_reason, native_break_taken, final_lp_flag,
%                        final_inner_converged, lp_failure_iters) are preserved
%                        beside it.  A failed subproblem is NEVER reported as
%                        convergence -- see LOCALSTOPPING below.
%     .objective_history maximized frequency per iteration [rad/s]
%     .last_obj          final objective
%     .path_identity     which implementation root executed (WP6 evidence)
%     .log               OLHOFFOPT's own event log, verbatim
%
%   See also REPRO2007_CONFIG, REPRO2007_PATHS, REPRO2007_HISTORY,
%            REPRO2007_LP_FLAGS, REPRO2007_REGRESSION, OLHOFFOPT.

initTic = tic;

if nargin < 1 || isempty(runCfg)
    runCfg = struct();
end
if ~isstruct(runCfg) || numel(runCfg) ~= 1
    error('run_repro2007:InvalidInput', 'runCfg must be a scalar struct.');
end

% ---- isolated path, asserted before anything else runs ------------------
guard = repro2007_paths(); %#ok<NASGU>  destroyed on return, restoring the path
pathIdentity = repro2007_assert_identity();

% ---- configuration ------------------------------------------------------
configName = localGet(runCfg, 'config', 'fig3a_best');
[cfg, configMeta] = repro2007_config(configName);

overrideApplied = false;
map = { ...
    'nelx',        'nelx'
    'nely',        'nely'
    'volfrac',     'volfrac'
    'target_mode', 'n'
    'rmin_elem',   'rminEl'
    'rmin_phys',   'rminPhys'
    'move',        'move'
    'tol_mult',    'tolMult'
    'max_outer',   'maxOuter'
    'tol_outer',   'tolOuter'
    'mass_interp', 'massInterp'
    'filter_mode', 'filterMode'
    'inner_solver','innerSolver'
    'off_diag',    'offDiag'
    'n_modes_max', 'Nmax'
    'penal',       'p'
    'rho_min',     'rhomin'
    'rho0',        'rho0'
    'E0',          'E'
    'nu',          'nu'
    'rho_m',       'rhom'
    'L',           'a'
    'H',           'b'
    'thickness',   't'
    'support',     'support'
    'axial',       'axial'
    'elem_type',   'elemType'
    'mass_type',   'massType'
    'eig_solver',  'solver'
    'threads',     'threads'
    'verbose',     'verbose'};

for i = 1:size(map, 1)
    if isfield(runCfg, map{i,1}) && ~isempty(runCfg.(map{i,1}))
        cfg.(map{i,2}) = runCfg.(map{i,1});
        overrideApplied = true;
    end
end

% rmin: the imported OLHOFFOPT lets rminPhys override rminEl.  If the caller
% supplied only rminEl, clear any physical radius the named configuration
% carried, otherwise the element radius would be silently discarded.
if isfield(runCfg, 'rmin_elem') && ~isempty(runCfg.rmin_elem) ...
        && ~(isfield(runCfg, 'rmin_phys') && ~isempty(runCfg.rmin_phys))
    cfg.rminPhys = [];
end

% Support code, following the repository's SS/CS/CC convention.
if isfield(runCfg, 'support_type') && ~isempty(runCfg.support_type)
    code = upper(char(string(runCfg.support_type)));
    switch code
        case 'SS', cfg.bc = 'a';   % simply supported both ends   (paper Fig. 2a)
        case 'CS', cfg.bc = 'b';   % clamped-simply supported     (paper Fig. 2b)
        case 'CC', cfg.bc = 'c';   % clamped both ends            (paper Fig. 2c)
        otherwise
            error('run_repro2007:UnsupportedBC', ...
                ['support_type "%s" is not available in the clean-room ' ...
                 'reproduction.  It implements the paper Fig. 2 beam only: ' ...
                 'SS, CS or CC.'], code);
    end
    overrideApplied = true;
end

if overrideApplied
    configMeta.baseline_artifact = '';
    configMeta.omega_expected = [];
    configMeta.is_reproduction = false;
end
cfg.name = configName;

approachName  = localGet(runCfg, 'approach_name', 'OlhoffDu2007Repro');
recordHistory = localGet(runCfg, 'record_history', true);
saveResults   = localGet(runCfg, 'save_results', false);

localValidate(cfg);

initTime = toc(initTic);

% ---- run ----------------------------------------------------------------
% OLHOFFOPT calls maxNumCompThreads(cfg.threads) and does not restore it.  That
% is deliberate in the reproduction (single-threaded BLAS is what its timings
% were measured against), but it would leak into whatever runs next in the same
% MATLAB session, so the previous value is restored afterwards.  The run itself
% is unaffected.
threadsBefore = maxNumCompThreads();
restoreThreads = onCleanup(@() maxNumCompThreads(threadsBefore));

if cfg.verbose
    fprintf('[%s] clean-room Du-Olhoff 2007 reproduction (Eq. 22 LP route)\n', approachName);
    fprintf('[%s] root   : %s\n', approachName, pathIdentity.root);
    fprintf('[%s] config : %s  (%dx%d, n=%d, rmin=%g el, move=%g, tolMult=%g, %s route)\n', ...
        approachName, configName, cfg.nelx, cfg.nely, cfg.n, cfg.rminEl, ...
        cfg.move, cfg.tolMult, upper(cfg.innerSolver));
end

loopTic = tic;
res = olhoffOpt(cfg);
loopTime = toc(loopTic);

postTic = tic;

% ---- outputs in repository convention -----------------------------------
x     = res.rho(:);
omega = localToVec3(res.omega(:));
nIter = res.nOuter;
if nIter > 0
    tIter = loopTime / nIter;
else
    tIter = NaN;
end

info = struct();
info.native        = res;
info.cfg           = res.cfg;
info.config_meta   = configMeta;
info.path_identity = pathIdentity;
info.log           = res.log;
info.mode_table    = res.modeTable;

if recordHistory
    info.history = repro2007_history(res);
else
    info.history = struct([]);
end

objectiveHistory = sqrt(max(res.hist.beta(:), 0));
info.objective_history = objectiveHistory;
if isempty(objectiveHistory)
    info.last_obj = NaN;
else
    info.last_obj = objectiveHistory(end);
end

info.timing = struct( ...
    'initialization_time',  initTime, ...
    'loop_time',            loopTime, ...
    'total_loop_time',      loopTime, ...
    'eigensolve_time',      sum(res.hist.tEig), ...
    'gradient_time',        sum(res.hist.tGrad), ...
    'subproblem_time',      sum(res.hist.tInner), ...
    'solver_wallclock',     res.wallclock, ...
    'postprocessing_time',  0);

% Outer/inner iteration counts.  This method has a genuine two-level structure
% -- the Fig. 1 outer loop, and the sub-optimization problem (25) solved inside
% each outer iteration -- and reporting only one of them hides where the cost
% goes.  For the Eq. (22) LP route `inner` is one linprog solve per outer
% iteration by construction; for the MMA route it is the accumulated count of
% MMA sub-iterates, which is where ~97% of that route's wall time sits
% (NOTES.md section 5).
if isempty(res.hist.cumInner)
    totalInner = NaN;
else
    totalInner = res.hist.cumInner(end);
end
info.iterations = struct( ...
    'outer',           nIter, ...
    'inner',           totalInner, ...
    'inner_per_outer', totalInner / max(nIter, 1), ...
    'inner_solver',    lower(char(cfg.innerSolver)));

info.stopping = repro2007_stopping(res, cfg);

info.timing.postprocessing_time = toc(postTic);

if cfg.verbose
    fprintf('[%s] done: %d outer iterations, %.1f s (%.3f s/iter)\n', ...
        approachName, nIter, loopTime, tIter);
    fprintf('[%s] omega = %.4f / %.4f / %.4f rad/s   gap = %.4f%%   N = %d\n', ...
        approachName, omega(1), omega(2), omega(3), ...
        100*(omega(2)-omega(1))/omega(1), res.hist.N(end));
    fprintf('[%s] stop  : %s  [%s]  (native: %s, lp_flag %g, inner_converged %d)\n', ...
        approachName, info.stopping.stop_reason, info.stopping.status, ...
        info.stopping.native_stop_reason, info.stopping.final_lp_flag, ...
        info.stopping.final_inner_converged);
    for i = 1:numel(res.log)
        fprintf('[%s] LOG: %s\n', approachName, res.log{i});
    end
end

if saveResults
    outDir = localGet(runCfg, 'results_dir', fullfile(repro2007_root(), 'results'));
    if exist(outDir, 'dir') ~= 7
        mkdir(outDir);
    end
    label = localGet(runCfg, 'run_label', configName);
    outFile = fullfile(outDir, [char(label) '.mat']);
    save(outFile, 'res', 'info', '-v7.3');
    if cfg.verbose
        fprintf('[%s] wrote %s\n', approachName, outFile);
    end
end
end

% -------------------------------------------------------------------------
function v = localGet(s, name, defaultValue)
if isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = defaultValue;
end
end

function v = localToVec3(w)
v = NaN(3, 1);
k = min(3, numel(w));
v(1:k) = w(1:k);
end

function localValidate(cfg)
if cfg.nelx < 1 || cfg.nely < 1
    error('run_repro2007:InvalidMesh', 'nelx and nely must be >= 1.');
end
if strcmpi(cfg.support, 'mid') && mod(cfg.nely, 2) ~= 0
    error('run_repro2007:OddNely', ...
        ['support = ''mid'' requires an even nely (got %d).  The paper''s ' ...
         'simple support sits at mid-height of the end face; an odd nely has ' ...
         'no node there.'], cfg.nely);
end
if cfg.n < 1
    error('run_repro2007:InvalidTargetMode', 'target_mode (n) must be >= 1.');
end
if cfg.n + cfg.Nmax < cfg.n + 1
    error('run_repro2007:InvalidNmax', 'n_modes_max (Nmax) must be >= 1.');
end
if cfg.volfrac <= 0 || cfg.volfrac > 1
    error('run_repro2007:InvalidVolfrac', 'volfrac must lie in (0, 1].');
end
if cfg.move <= 0
    error('run_repro2007:InvalidMove', 'move must be > 0.');
end
if ~any(strcmpi(cfg.innerSolver, {'lp', 'mma'}))
    error('run_repro2007:InvalidInnerSolver', ...
        'inner_solver must be ''lp'' or ''mma'' (got "%s").', cfg.innerSolver);
end
if ~any(strcmpi(cfg.filterMode, {'diag', 'all', 'none'}))
    error('run_repro2007:InvalidFilterMode', ...
        'filter_mode must be ''diag'', ''all'' or ''none'' (got "%s").', cfg.filterMode);
end
if ~any(strcmpi(cfg.massInterp, {'4', '4a', '4b', 'lin'}))
    error('run_repro2007:InvalidMassInterp', ...
        'mass_interp must be ''4'', ''4a'', ''4b'' or ''lin'' (got "%s").', cfg.massInterp);
end
if strcmpi(cfg.innerSolver, 'lp') && exist('linprog', 'file') ~= 2
    error('run_repro2007:MissingLinprog', ...
        ['The Eq. (22) LP route needs linprog (Optimization Toolbox), which ' ...
         'is not available.  Set inner_solver = ''mma'' to use the paper''s ' ...
         'labelled alternative, noting that it does not converge once N >= 2.']);
end
end
