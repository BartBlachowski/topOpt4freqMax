function r3_w3_regenerate_old(oldRepo)
%R3_W3_REGENERATE_OLD  Reviewer 3, W3: regenerate the clamped-beam designs of the SUBMITTED
%   version (Table 2 of the submission: omega1 = 275.28 rad/s at alpha = 1 and 347.70 rad/s
%   at alpha = 0.75) with the OLD research code, commit a2261f7 of the research repository
%   (examples/ClampedBeam/BeamTopOptFreq.json: 400x50, MMA, filter 2 elements, move 0.2,
%   tol 1e-3, E_min = 1e-6 E0, rho_min = 1e-6).  Plots are switched off; nothing else changed.
%   oldRepo: checkout of a2261f7 (git worktree add --detach <dir> a2261f7).
%   Output: results/r3/w3_old_designs.mat (x in the old code's element numbering, omega).
out = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results', 'r3');
maxNumCompThreads(1);
addpath(fullfile(oldRepo, 'tools', 'Matlab'), fullfile(oldRepo, 'ourApproach', 'Matlab'), fullfile(oldRepo, 'examples'));
data = jsondecode(fileread(fullfile(oldRepo, 'examples', 'ClampedBeam', 'BeamTopOptFreq.json')));
data.postprocessing.visualize_live = false;
data.postprocessing.visualize_modes.enabled = false;
data.postprocessing.visualize_topology_modes.enabled = false;
data.postprocessing.save_final_image = false;
data.postprocessing.save_frequency_iterations = false;
alphas = [1 0.75];  old = struct('alpha', {}, 'x', {}, 'omega', {}, 'nIter', {}, 'tIter', {});
here = pwd;  tmp = tempname;  mkdir(tmp);  cd(tmp);  c = onCleanup(@() cd(here));
for a = alphas
    data.domain.load_cases(1).factor = a;  data.domain.load_cases(2).factor = 1 - a;
    [x, omega, tIter, nIter] = run_topopt_from_json(data);
    fprintf('alpha = %.2f: %d iterations, omega = %s rad/s\n', a, nIter, mat2str(omega(:)', 6));
    old(end+1) = struct('alpha', a, 'x', x, 'omega', omega, 'nIter', nIter, 'tIter', tIter); %#ok<AGROW>
    save(fullfile(out, 'w3_old_designs.mat'), 'old');
end
end
