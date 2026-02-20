clear; clc;
close all;

% Load base beam configuration (simply supported beam, 8x1 m)
jsonPath = fullfile(fileparts(mfilename('fullpath')), 'BeamTopOptFreq.json');
data = jsondecode(fileread(jsonPath));

% Disable visualization and image saving for clean performance measurement
data.optimisation.visualise_live   = 'no';
data.optimisation.save_final_image = 'yes';
data.optimisation.save_snapshot_image = 'no';

% Fix filter radius to 2 finite elements regardless of resolution.
% The base JSON uses physical units (0.04 m), which gives < 1 element at
% coarser meshes and causes checkerboard patterns.  Switching to 'element'
% units with radius = 2 keeps the filter consistent across all resolutions.
data.optimisation.filter.radius       = 2;
data.optimisation.filter.radius_units = 'element';

% -------------------------------------------------------------------------
% Resolutions: those from Table 1 in the paper (160x20, 240x30, 320x40)
% plus two additional ones (240x30 already in paper; 400x50 is new)
% -------------------------------------------------------------------------
resolutions = [
    160,  20;
    240,  30;
    320,  40;
    400,  50;
    600,  75;
];
nRes = size(resolutions, 1);

% Methods to compare
approaches   = {'Olhoff',         'Yuksel',         'OurApproach'       };
methodLabels = {'OlhoffApproach', 'YukselApproach', 'ProposedApproach'  };
nMethods     = numel(approaches);

% Storage: rows = resolutions, columns = methods
omega_all  = NaN(nRes, nMethods);
tIter_all  = NaN(nRes, nMethods);
nIter_all  = NaN(nRes, nMethods);
mem_all    = NaN(nRes, nMethods);

% -------------------------------------------------------------------------
% Run all (resolution × method) combinations
% -------------------------------------------------------------------------
for r = 1:nRes
    data.domain.mesh.nelx = resolutions(r, 1);
    data.domain.mesh.nely = resolutions(r, 2);

    for m = 1:nMethods
        data.optimisation.approach = approaches{m};
        fprintf('Running %-18s  mesh %4dx%-3d ...\n', ...
            methodLabels{m}, resolutions(r,1), resolutions(r,2));

        [~, omega, tIter, nIter, mem] = run_topopt_from_json(data);

        omega_all(r, m) = omega(1);          % first natural frequency [rad/s]
        tIter_all(r, m) = tIter;             % avg time per iteration [s/iter]
        nIter_all(r, m) = nIter;             % number of iterations
        mem_all(r, m)   = mem;               % peak RAM [MB]
    end
end

% Total run time = average iteration time × number of iterations
tTotal_all = tIter_all .* nIter_all;

% -------------------------------------------------------------------------
% Print performance table (mirrors Table 1 from Yuksel et al.)
% -------------------------------------------------------------------------
sepWidth = 92;
sep = repmat('-', 1, sepWidth);

fprintf('\n');
fprintf('Table 1. Run time comparison between methods for maximizing the first\n');
fprintf('natural frequency of a simply supported beam (8 m x 1 m, vf = 0.5).\n');
fprintf('\n');
fprintf('%-20s  %-9s  %16s  %20s  %18s\n', ...
    'Method', 'Mesh size', 'Run time (s)', 'Run time/iter (s/iter)', 'Max RAM (MB)');
fprintf('%s\n', sep);

for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));

    for m = 1:nMethods
        if isnan(tTotal_all(r,m))
            timeStr   = 'N/A';
            iterStr   = 'N/A';
            ramStr    = 'N/A';
        else
            timeStr = sprintf('%.1f', tTotal_all(r,m));
            iterStr = sprintf('%.2f', tIter_all(r,m));
            ramStr  = sprintf('%.0f', mem_all(r,m));
        end
        fprintf('%-20s  %-9s  %16s  %20s  %18s\n', ...
            methodLabels{m}, meshStr, timeStr, iterStr, ramStr);
    end

    if r < nRes
        fprintf('%s\n', sep);
    end
end

fprintf('%s\n', sep);
fprintf('\n');

% -------------------------------------------------------------------------
% Also print achieved natural frequencies for reference
% -------------------------------------------------------------------------
fprintf('Achieved first natural frequency omega_1 [rad/s]:\n');
fprintf('%s\n', sep);
fprintf('%-20s  %-9s  %16s\n', 'Method', 'Mesh size', 'omega_1 (rad/s)');
fprintf('%s\n', sep);

for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
    for m = 1:nMethods
        if isnan(omega_all(r,m))
            omStr = 'N/A';
        else
            omStr = sprintf('%.1f', omega_all(r,m));
        end
        fprintf('%-20s  %-9s  %16s\n', methodLabels{m}, meshStr, omStr);
    end
    if r < nRes
        fprintf('%s\n', sep);
    end
end
fprintf('%s\n', sep);
