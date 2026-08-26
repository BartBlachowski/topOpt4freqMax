function H = repro2007_history(res)
%REPRO2007_HISTORY  Per-outer-iteration history in the repository schema.
%
%   H = REPRO2007_HISTORY(res) reshapes what OLHOFFOPT recorded in res.hist
%   into the columns the performance comparison needs, without recomputing
%   anything and without re-running the solver.  Every column is either copied
%   from res.hist or derived from it algebraically.
%
%   COLUMNS (all 1 x nOuter unless stated)
%   --------------------------------------
%   Identity
%     iter              outer iteration number, 1-based
%     stage             1 throughout; this method is single-stage.  Present so
%                       the column exists for methods that are not (Yuksel).
%     stage_iter        same as iter, for the same reason
%
%   Spectrum -- the requested modes, not a summary of them
%     omega             (J+1) x nOuter, ALL modes OLHOFFOPT computed each
%                       iteration, J = n + Nmax.  Kept in full because the
%                       multiplicity question is about which modes are close,
%                       and a three-mode summary cannot answer it.
%     omega1, omega2, omega3
%                       the first three, extracted for convenience
%     n_modes_recorded  size(omega, 1)
%
%   Multiplicity -- BOTH the detected count and the gap it was detected from
%     N                 multiplicity detected by OLHOFFOPT at this iteration
%     gap_rel           ACTUAL relative eigengap (omega_{n+1} - omega_n)/omega_n
%     gap_abs           actual absolute eigengap [rad/s]
%     bimodal           N >= 2, recorded as a convenience only
%     mult_tolerance    the tolerance N was decided against (scalar, in .meta)
%     multJ             true where omega_J was itself multiple, i.e. where the
%                       paper's (25b) is undefined.  OLHOFFOPT logs these and
%                       applies no patch; the column makes them countable.
%
%     N and gap_rel are BOTH recorded on purpose.  N is a thresholded view of
%     gap_rel at cfg.tolMult, so reporting N alone would make the reported
%     multiplicity a function of an unstated tolerance -- which is exactly the
%     ambiguity this reproduction exists to expose.
%
%   Objective and design
%     objective         sqrt(beta): the maximized frequency in rad/s.  beta is
%                       the bound variable of problem (25) and lives in
%                       eigenvalue units; the square root puts it in the same
%                       units as omega so the two are directly comparable.
%     beta              the raw bound variable, eigenvalue units
%     vol               mean(rho) after the update
%     rV                |mean(rho) - volfrac| / volfrac
%     d_inf             max_e |drho_e|, the outer-iteration density increment
%     move_limit        the move limit in force (scalar, in .meta)
%     move_saturated    d_inf equals the move limit to machine precision
%
%   Subproblem
%     n_inner           inner/subproblem solves this outer iteration.  1 for
%                       the LP route; up to cfg.maxInner MMA sub-iterates for
%                       the MMA route.
%     cum_inner         running total
%     inner_converged   the subproblem's own convergence status
%     lp_flag           linprog exit flag where the LP route ran, recovered
%                       from res.log for failed solves and 1 for successful
%                       ones; NaN for the MMA route
%     degen_hits        times the (25d) subeigenvalue matrix was degenerate,
%                       where the gradients are not uniquely defined
%
%   Runtime
%     t_eig, t_grad, t_inner    per-iteration component times [s]
%     elapsed_s                 cumulative solver time [s]
%
%   NOT AVAILABLE -- stated rather than silently left NaN
%   ----------------------------------------------------
%   grayness and d_rms are NaN for every iteration except the last.
%
%   Both are functions of the full density field, and the imported OLHOFFOPT
%   records only mean(rho) per iteration -- it keeps no per-iteration density
%   history.  Producing them would require adding a density recorder to
%   algo/olhoffOpt.m, and that file is deliberately held byte-identical to the
%   clean-room source (SOURCE_SHA256.txt) for the duration of the paper
%   revision.  The final-iterate values are recorded in
%   info.stopping.final_grayness and are what the performance comparison
%   actually reads.  See MIGRATION_REPRODUCTION2007_REPORT.md.
%
%   See also RUN_REPRO2007, TOPOPT_HISTORY_INIT.

h = res.hist;
cfg = res.cfg;
n = numel(h.N);

H = struct();
H.iter       = 1:n;
H.stage      = ones(1, n);
H.stage_iter = 1:n;

% ---- spectrum -----------------------------------------------------------
H.omega = h.omega;
H.n_modes_recorded = size(h.omega, 1);
H.omega1 = localRow(h.omega, 1, n);
H.omega2 = localRow(h.omega, 2, n);
H.omega3 = localRow(h.omega, 3, n);

% ---- multiplicity: the count AND the gap it came from -------------------
H.N = h.N(:).';
nTarget = cfg.n;
if size(h.omega, 1) >= nTarget + 1
    wn  = h.omega(nTarget, :);
    wn1 = h.omega(nTarget + 1, :);
    H.gap_abs = wn1 - wn;
    H.gap_rel = (wn1 - wn) ./ wn;
else
    H.gap_abs = NaN(1, n);
    H.gap_rel = NaN(1, n);
end
H.bimodal = H.N >= 2;
H.multJ   = logical(h.multJ(:).');

% ---- objective and design ----------------------------------------------
H.beta      = h.beta(:).';
H.objective = sqrt(max(h.beta(:).', 0));
H.vol       = h.vol(:).';
H.rV        = abs(H.vol - cfg.volfrac) / cfg.volfrac;
H.d_inf     = h.dxOuter(:).';
H.move_saturated = abs(H.d_inf - cfg.move) < 1e-12;

% ---- subproblem ---------------------------------------------------------
H.n_inner         = h.nInner(:).';
H.cum_inner       = h.cumInner(:).';
H.inner_converged = logical(h.innerConv(:).');
H.degen_hits      = h.degen(:).';
H.lp_flag         = localLpFlags(res, n);

% ---- runtime ------------------------------------------------------------
H.t_eig   = h.tEig(:).';
H.t_grad  = h.tGrad(:).';
H.t_inner = h.tInner(:).';
H.elapsed_s = cumsum(H.t_eig + H.t_grad + H.t_inner);

% ---- deliberately unavailable per iteration (see header) ----------------
H.grayness = NaN(1, n);
H.d_rms    = NaN(1, n);
if n > 0
    rho = res.rho(:);
    H.grayness(n) = mean(4 * rho .* (1 - rho));
end

% ---- run-level context --------------------------------------------------
H.meta = struct( ...
    'method',               'OlhoffDu2007Repro', ...
    'implementation',       'Du-Olhoff 2007 clean-room reproduction (Eq. 22 LP route)', ...
    'objective_definition', ['maximized circular frequency omega_n = sqrt(beta) ' ...
                             '[rad/s]; larger is better'], ...
    'target_mode',          cfg.n, ...
    'mult_tolerance',       cfg.tolMult, ...
    'move_limit',           cfg.move, ...
    'volfrac',              cfg.volfrac, ...
    'rmin_elem',            cfg.rminEl, ...
    'filter_mode',          cfg.filterMode, ...
    'mass_interp',          cfg.massInterp, ...
    'inner_solver',         cfg.innerSolver, ...
    'off_diag_coupling',    logical(cfg.offDiag), ...
    'nelx',                 cfg.nelx, ...
    'nely',                 cfg.nely, ...
    'n_outer',              n, ...
    'grayness_note',        ['per-iteration grayness and d_rms unavailable: ' ...
                             'olhoffOpt.m is held byte-identical to the ' ...
                             'clean-room source and records no density history'], ...
    'markers',              struct('iter', {}, 'kind', {}, 'detail', {}));

% Multiplicity transitions are this method's structural events, so they are
% recorded the way the shared recorder records stage handoffs.
if n > 0
    changes = [true, diff(H.N) ~= 0];
    idx = find(changes);
    for i = 1:numel(idx)
        k = idx(i);
        if k == 1
            detail = sprintf('N = %d at start', H.N(k));
        else
            detail = sprintf('N %d -> %d', H.N(k-1), H.N(k));
        end
        H.meta.markers(end+1) = struct('iter', k, 'kind', 'multiplicity', ...
            'detail', detail);
    end
end
H.k_mult = 0;
firstBimodal = find(H.N >= 2, 1);
if ~isempty(firstBimodal)
    H.k_mult = firstBimodal;
end
end

% -------------------------------------------------------------------------
function r = localRow(A, k, n)
if size(A, 1) >= k
    r = A(k, :);
else
    r = NaN(1, n);
end
end

function flags = localLpFlags(res, n)
%LOCALLPFLAGS  Recover the linprog exit flag per iteration.
%
%   OLHOFFOPT stores the subproblem's boolean convergence status in
%   hist.innerConv and writes the numeric flag into res.log only when the solve
%   fails.  Both are therefore needed to reconstruct the per-iteration flag,
%   and neither requires modifying the imported solver.

if ~strcmpi(res.cfg.innerSolver, 'lp')
    flags = NaN(1, n);
    return
end

conv = logical(res.hist.innerConv(:).');
flags = NaN(1, n);
flags(conv) = 1;                 % innerLoopLP sets conv = (flag == 1)

for i = 1:numel(res.log)
    tok = regexp(res.log{i}, ...
        '^iter (\d+): LP inner solve failed \(flag=(-?\d+)\)', 'tokens', 'once');
    if isempty(tok)
        continue
    end
    k = str2double(tok{1});
    if k >= 1 && k <= n
        flags(k) = str2double(tok{2});
    end
end
end
