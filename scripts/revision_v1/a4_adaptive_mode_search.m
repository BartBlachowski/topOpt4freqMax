function result = a4_adaptive_mode_search(Kf, Mf, free, ndof, xPhys, ctx, phiPrev, phi0, opts)
%A4_ADAPTIVE_MODE_SEARCH  Deterministic A4 Phase-2 ladder search.
% Implements A4_RECOVERY_PHASE2_SPECIFICATION.md §§3.1--3.7. Every successful
% search performs a confirmation expansion, and all returned quantities come
% from the widest solve. The optional solve_modes function handle exists only
% to permit deterministic validation fixtures; production calls omit it.

if nargin < 9, opts = struct(); end
c = a4_phase2_constants();
solveModes = localOpt(opts, 'solve_modes', []);
result = localBlankResult();
result.constants = c;

if localHasNonfinite(Kf) || localHasNonfinite(Mf) || any(~isfinite(xPhys(:))) || ...
        any(~isfinite(phiPrev(:))) || any(~isfinite(phi0(:)))
    result.search_outcome = 'SOLVER_FAILURE';
    result.failure_message = 'NaN or Inf in search input';
    return;
end

priorSelectedPhi = [];
priorRungHadSelection = false;
for iRung = 1:numel(c.window_ladder)
    m = c.window_ladder(iRung);
    result.window_rungs_solved(end+1) = m; %#ok<AGROW>
    result.eigensolve_count = result.eigensolve_count + 1;
    try
        if isempty(solveModes)
            [omegas, Phi, residuals] = localSolve(Kf, Mf, free, ndof, m, c.eigensolver);
        else
            [omegas, Phi, residuals] = solveModes(m);
        end
        if numel(omegas) ~= m || size(Phi, 2) ~= m || numel(residuals) ~= m
            error('a4_adaptive_mode_search:WrongSolveSize', ...
                'Window %d returned %d eigenvalues, %d vectors, %d residuals.', ...
                m, numel(omegas), size(Phi,2), numel(residuals));
        end
        if any(~isfinite(omegas(:))) || any(~isfinite(Phi(:))) || any(~isfinite(residuals(:)))
            error('a4_adaptive_mode_search:NonfiniteSolve', ...
                'Window %d contains NaN or Inf.', m);
        end
        screen = a4_mode_screen(Phi, omegas, xPhys, ctx, phiPrev, phi0);
        for k = 1:m
            screen.modes(k).eigensolver_status = sprintf( ...
                'converged; residual=%.17g', residuals(k));
        end
        localAssertFiniteScreen(screen);
    catch ME
        result.search_outcome = 'SOLVER_FAILURE';
        result.m_final = m;
        result.failure_identifier = ME.identifier;
        result.failure_message = ME.message;
        return;
    end

    % Replace the result on every rung: §3.5 requires final-window values.
    result.m_final = m;
    result.omegas = omegas(:);
    result.Phi = Phi;
    result.residuals = residuals(:);
    result.screen = screen;
    result.candidates = screen.modes;
    result.n_candidates = m;
    result.n_admissible = sum([screen.modes.admissible]);
    result.selected_index = screen.selected;
    result.tie_flag = screen.tieOccurred;

    if screen.selected == 0
        priorSelectedPhi = [];
        priorRungHadSelection = false;
        if m == c.M_max
            result.search_outcome = 'REFERENCE_UNAVAILABLE';
            result.stability_flag = 'n/a';
            return;
        end
        continue;
    end

    selectedPhi = Phi(:, screen.selected);
    if priorRungHadSelection
        result.stability_mac = a4_mac(selectedPhi, priorSelectedPhi, ctx.M);
        if result.stability_mac >= c.tau_stab
            result.search_outcome = 'SELECTED';
            result.stability_flag = 'confirmed';
            return;
        end
    end

    if m == c.M_max
        result.search_outcome = 'SELECTED';
        result.stability_flag = 'unconfirmed';
        return;
    end
    priorSelectedPhi = selectedPhi;
    priorRungHadSelection = true;
end
end

function [omegas, Phi, residuals] = localSolve(Kf, Mf, free, ndof, m, settings)
nFree = size(Kf, 1);
if nFree <= m
    error('a4_adaptive_mode_search:InsufficientDofs', ...
        'Window %d requires more than %d free DOFs.', m, nFree);
end
eigOpts = struct('disp', settings.display, 'maxit', settings.max_iterations, ...
    'tol', settings.tolerance);
v0 = (1:nFree)';
eigOpts.v0 = v0 / norm(v0);
[V, D, flag] = eigs(Kf, Mf, m, settings.sigma, eigOpts);
if flag ~= 0
    error('a4_adaptive_mode_search:Nonconvergence', ...
        'eigs returned non-zero convergence flag %d for window %d.', flag, m);
end
lam = real(diag(D));
[lam, order] = sort(lam, 'ascend');
V = real(V(:, order));
if any(lam <= 0)
    error('a4_adaptive_mode_search:NonpositiveEigenvalue', ...
        'The requested spectrum contains a non-positive eigenvalue.');
end
omegas = sqrt(lam);
Phi = zeros(ndof, m);
residuals = NaN(m, 1);
for k = 1:m
    vk = mass_normalize_modes(V(:, k), Mf);
    vk = orient_modes_deterministic(vk);
    denom = norm(Kf*vk) + abs(lam(k))*norm(Mf*vk);
    residuals(k) = norm(Kf*vk - lam(k)*(Mf*vk)) / max(denom, realmin);
    Phi(free, k) = vk;
end
end

function localAssertFiniteScreen(screen)
for k = 1:numel(screen.modes)
    row = screen.modes(k);
    vals = [row.omega, row.mac_prev, row.mac_phi0, row.mac_solid, ...
        row.largest_support_component_kinetic_fraction, ...
        row.low_density_strain_fraction, row.low_density_kinetic_fraction];
    if any(~isfinite(vals))
        error('a4_adaptive_mode_search:NonfiniteScreen', ...
            'Candidate %d contains a non-finite screen quantity.', k);
    end
end
end

function tf = localHasNonfinite(A)
tf = any(~isfinite(nonzeros(A)));
end

function v = localOpt(s, name, default)
v = default;
if isstruct(s) && isfield(s, name) && ~isempty(s.(name)), v = s.(name); end
end

function r = localBlankResult()
r = struct('search_outcome', '', 'stability_flag', 'n/a', ...
    'stability_mac', NaN, 'window_rungs_solved', [], 'm_final', 0, ...
    'eigensolve_count', 0, 'n_candidates', 0, 'n_admissible', 0, ...
    'selected_index', 0, 'tie_flag', false, 'omegas', [], 'Phi', [], ...
    'residuals', [], 'screen', struct(), 'candidates', [], ...
    'failure_identifier', '', 'failure_message', '', 'constants', struct());
end
