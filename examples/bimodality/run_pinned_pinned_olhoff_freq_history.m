%RUN_PINNED_PINNED_OLHOFF_FREQ_HISTORY  Du-Olhoff omega_1..3 history, pinned-pinned beam.
%
%   Reconstructs the "Olhoff frequency history" figure (paper Fig. 9a) for the
%   simply-supported (pinned-pinned) beam of Du & Olhoff (2007), Fig. 2a
%   (docs/s00158-007-0101-y.pdf): 8 x 1 domain, 240 x 30 mesh, volume
%   fraction 0.5, both ends pinned at mid-height (the analysis/Olhoff
%   defaults domain.boundary.condition = simplySupported, support = midHeight).
%
%   Uses the production preset (olhoffcurrent_production_preset, currently
%   duOlhoffPedersenAdaptiveBoxSensitivityFiltered), i.e. exactly the
%   configuration of the Du-Olhoff rows of the paper's Table 1 (111 outer
%   iterations at 240x30).  With this preset omega_1 and omega_2 run together
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
%   Output (next to this file):
%     Olhoff_240x30_freq_iterations.fig / .png

nelx = 240;
nely = 30;

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo, 'analysis', 'Olhoff'));
addpath(fullfile(repo, 'tools', 'Matlab'));

guard = olhoffcurrent_paths(); %#ok<NASGU>  keep the fail-closed path guard alive
preset = olhoffcurrent_production_preset().name;
cfg = olhoffcurrent_config(nelx, nely, 'Preset', preset);
res = olhoffSolve(cfg);

omegaHist = double(res.hist.omega).';        % nOuter x Jcalc, rad/s
save_frequency_iteration_plot(omegaHist, 'Olhoff', nelx, nely, here);

w = double(res.omega(:));
fprintf('Olhoff (%s) pinned-pinned %dx%d: %d outer iterations, omega_1..3 = %.2f, %.2f, %.2f rad/s\n', ...
    preset, nelx, nely, size(omegaHist, 1), w(1), w(2), w(3));
