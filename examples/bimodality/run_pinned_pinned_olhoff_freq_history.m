%RUN_PINNED_PINNED_OLHOFF_FREQ_HISTORY  Du-Olhoff omega_1..3 history, pinned-pinned beam.
%
%   Reconstructs the "Olhoff frequency history" figure (paper Fig. 9a) for the
%   simply-supported (pinned-pinned) beam of Du & Olhoff (2007), Fig. 2a
%   (docs/s00158-007-0101-y.pdf): 8 x 1 domain, 240 x 30 mesh, volume
%   fraction 0.5, both ends pinned at mid-height (the analysis/Olhoff
%   defaults domain.boundary.condition = simplySupported, support = midHeight).
%
%   Uses the production preset (olhoffcurrent_production_preset, currently
%   duOlhoffPedersenAdaptiveBoxSensitivityFiltered); with the default stop
%   parameters below this is exactly the configuration of the Du-Olhoff rows
%   of the paper's Table 1 (111 outer iterations at 240x30).  With this preset omega_1 and omega_2 run together
%   over roughly outer iterations 8-25 and then separate (final 167.3 / 187.2
%   rad/s): the converged design is not bimodal at this mesh.  (At 400x50:
%   93 outer, 166.5 / 198.1 rad/s.  The historical SIMP / eq. (4b) preset
%   duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered stays closer to
%   bimodal there, 162.9 / 175.5 rad/s in 139 outer iterations, but does not
%   match Table 1.)
%
%   The plotted omega_1..3 are the solver's own eigenfrequencies (Pedersen
%   stiffness, linear mass), i.e. the eigenvalues the bound formulation acts
%   on.  They are not re-evaluated with the common evaluator E1 used for the
%   proposed method in Fig. 9b: on transitional iterates E1 resolves clusters
%   of void-localized modes (e.g. 149-177 rad/s at outer iteration 32 of this
%   run) and its classifier then admits hybrid modes, producing a spurious
%   one-iterate drop of omega_3.  At convergence the two models agree to
%   within 1 % (native 167.34 / 187.16 / 312.51 rad/s, E1 167.33 / 186.91 /
%   309.19 rad/s).
%
%   olhoffcurrent_run does not return the per-iteration history, so this
%   runner takes the same canonical route it does (path guard -> named preset
%   config -> olhoffSolve) and reads res.hist.omega.
%
%   Stop parameters.  The run stops when ||drho||_2 < eps (Du & Olhoff sec.
%   3.5, Fig. 1).  The paper gives no value for eps; analysis/Olhoff uses
%   eps = 0.05*sqrt(NE/3200) (olh.config.epsilonForMesh), i.e. the same RMS
%   design change on every mesh.  stop_c replaces the 0.05 and max_iter caps
%   the outer iterations (a run that reaches the cap is CAP_HIT, not
%   converged).  stop_c = 0.2 stops at outer iteration 52 at 240x30, after
%   omega_1 has settled; at 160x20 stop_c >= 0.225 stops before the topology
%   has settled.
%
%   stop_criterion selects the rule; each reads only its own tolerance:
%     'l2_change'           the rule above, ||drho||_2 < stop_c*sqrt(NE/3200)
%     'max_change'          the proposed method's: max|drho| <= stop_tol on the
%                           design variable (olhoffcurrent_config
%                           'StopMaxChangeTolerance')
%     'relative_l2_change'  ||drho||_2/||rho||_2 < stop_rel_tol, rho the design
%                           before the update ('StopRelativeChangeTolerance')
%   The last two use no mesh scaling and no guards.
%
%   Every save_every_it-th iterate (and the last one) is also rendered, so the
%   topology can be compared with the frequency plateau of the history; the
%   snapshot of outer iteration k is the design whose omega_1..3 are plotted
%   at k (the design analysed at the start of iteration k).  olhoffSolve keeps
%   no design history, so the run switches on its per-iteration diagnostic
%   recorder (runtime.diagnostics, bitwise inert on the trajectory) and the
%   designs are rebuilt by replaying the solver's own update on the recorded
%   increments; the replay must reproduce res.rho exactly.
%
%   Output (next to this file):
%     Olhoff_240x30_freq_iterations.fig / .png
%     topologies/Olhoff_240x30_it<k>.png

nelx = 800;
nely = 100;
save_every_it = 25;   % topology snapshot every save_every_it iterations (+ last); 0 = none
stop_criterion = 'relative_l2_change';   % 'l2_change' (Table 1 rule) | 'max_change' | 'relative_l2_change'
stop_c = 0.08;        % stop when ||drho||_2 < stop_c*sqrt(NE/3200); 0.05 = Table 1  ('l2_change')
stop_tol = 0.02;      % stop when max|drho| <= stop_tol                              ('max_change')
stop_rel_tol = 1e-3;  % stop when ||drho||_2/||rho||_2 < stop_rel_tol; no calibrated value ('relative_l2_change')
max_iter = [];        % outer-iteration cap; [] = preset default (400)

% Release the path guard of an earlier run in this session first: overwriting
% it would restore its saved path AFTER the new guard installed, removing the
% solver from the path.
clear guard

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo, 'analysis', 'Olhoff'));
addpath(fullfile(repo, 'tools', 'Matlab'));

% The MATLAB path is session state: adding the repository with subfolders puts
% development/ (every historical Olhoff tree) on it, and the guard below then
% refuses to run.  Remove what the session handed us first, as
% examples/Performance/performance_comparison.m does; the guard re-checks.
pathScrub = olhoffcurrent_scrub_forbidden_paths(repo);
if ~isempty(pathScrub)
    fprintf('Removed %d non-production Olhoff path entries inherited from this MATLAB session.\n', ...
        numel(pathScrub));
end
guard = olhoffcurrent_paths(); %#ok<NASGU>  keep the fail-closed path guard alive
preset = olhoffcurrent_production_preset().name;
cfgArgs = {'Preset', preset, 'Diagnostics', save_every_it > 0};
if ~isempty(max_iter)
    cfgArgs = [cfgArgs, {'MaxOuter', max_iter}];
end
switch stop_criterion
    case 'l2_change'
        cfgArgs = [cfgArgs, {'StopToleranceFactor', stop_c}];   % eps = stop_c*sqrt(NE/3200)
        stopDesc = sprintf('stop_c = %g (||drho||_2 < %.4g)', stop_c, stop_c*sqrt(nelx*nely/3200));
    case 'max_change'
        cfgArgs = [cfgArgs, {'StopMaxChangeTolerance', stop_tol}];
        stopDesc = sprintf('stop_tol = %g (max|drho| <= %g)', stop_tol, stop_tol);
    case 'relative_l2_change'
        cfgArgs = [cfgArgs, {'StopRelativeChangeTolerance', stop_rel_tol}];
        stopDesc = sprintf('stop_rel_tol = %g (||drho||_2/||rho||_2 < %g)', stop_rel_tol, stop_rel_tol);
    otherwise
        error(['stop_criterion must be ''l2_change'', ''max_change'' or ' ...
               '''relative_l2_change'' (got ''%s'').'], stop_criterion);
end
cfg = olhoffcurrent_config(nelx, nely, cfgArgs{:});
res = olhoffSolve(cfg);

omegaHist = double(res.hist.omega).';        % nOuter x Jcalc, rad/s
save_frequency_iteration_plot(omegaHist, 'Olhoff', nelx, nely, here);

if save_every_it > 0
    % Replay of olhoffSolve's step 4 without projection:
    % rho_1 = design.initial, rho_{k+1} = min(1, max(rhomin, rho_k + drho_k)).
    assert(~cfg.projection.enabled, ...
        'The design replay assumes the unprojected update (projection.enabled = false).');
    nOuter = size(omegaHist, 1);
    assert(numel(res.diag.drho) == nOuter, 'Diagnostic record does not cover every outer iteration.');
    rho = cfg.design.initial*ones(numel(res.rho), 1);
    rhoHist = zeros(numel(rho), nOuter);
    for k = 1:nOuter
        rhoHist(:,k) = rho;
        rho = min(1, max(cfg.design.minimum, rho + res.diag.drho{k}));
    end
    assert(isequal(rho, res.rho), 'Replayed design history does not reproduce res.rho.');
    % model2D numbers element rows top-down (top88); the renderer expects bottom-up.
    rhoHist = reshape(flip(reshape(rhoHist, nely, nelx, nOuter), 1), nelx*nely, nOuter);
    save_topology_snapshots(rhoHist, omegaHist, save_every_it, ...
        'Olhoff', nelx, nely, fullfile(here, 'topologies'));
end

w = double(res.omega(:));
fprintf(['Olhoff (%s) pinned-pinned %dx%d, %s: %s after %d outer ' ...
    'iterations, omega_1..3 = %.2f, %.2f, %.2f rad/s\n'], ...
    preset, nelx, nely, stopDesc, res.status, size(omegaHist, 1), w(1), w(2), w(3));
