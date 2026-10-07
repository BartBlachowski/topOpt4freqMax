function pngPaths = save_topology_snapshots(xHist, omegaHist, saveEvery, approachName, nelx, nely, outDir)
%SAVE_TOPOLOGY_SNAPSHOTS  Topology image of every saveEvery-th iterate.
%
%   pngPaths = SAVE_TOPOLOGY_SNAPSHOTS(xHist, omegaHist, saveEvery, approachName, nelx, nely, outDir)
%
%   xHist        : n_e x nIter element densities; column k is the design
%                  ANALYSED at iteration k (element order e = ey + ex*nely + 1).
%   omegaHist    : nIter x k frequencies [rad/s] of those designs, i.e. the
%                  curves of SAVE_FREQUENCY_ITERATION_PLOT; omega_1 and omega_2
%                  of iteration k go into the title of snapshot k so that each
%                  image can be matched to its point on the frequency plot.
%                  Pass [] to omit them.
%   saveEvery    : snapshot cadence; iterations saveEvery, 2*saveEvery, ...
%                  plus the last iteration are written.  0 (or Inf) disables.
%   approachName : title and file-name prefix.
%   outDir       : destination folder, created when missing.
%
%   Files are <name>_<nelx>x<nely>_it<k>.png, k zero-padded so that they sort
%   by iteration.  Snapshots of the same approach and mesh left by an earlier
%   run are deleted first, so the folder never mixes two runs or cadences.
%   Rendering goes through renderTopologyDensity (black = solid).
%   Presentation only: call it after the solve.

    pngPaths = {};
    if ~(isnumeric(saveEvery) && isscalar(saveEvery) && saveEvery >= 0 && ...
            (mod(saveEvery, 1) == 0 || isinf(saveEvery)))
        error('save_topology_snapshots:InvalidCadence', ...
            'saveEvery must be a non-negative integer (0 or Inf disables).');
    end
    nIter = size(xHist, 2);
    if saveEvery == 0 || isinf(saveEvery) || nIter == 0
        return;
    end
    if size(xHist, 1) ~= nelx*nely
        error('save_topology_snapshots:InvalidField', ...
            'xHist must have nelx*nely = %d rows (got %d).', nelx*nely, size(xHist, 1));
    end
    if ~isempty(omegaHist) && size(omegaHist, 1) ~= nIter
        error('save_topology_snapshots:HistoryMismatch', ...
            'omegaHist has %d rows but xHist has %d iterations.', size(omegaHist, 1), nIter);
    end

    iters = unique([saveEvery:saveEvery:nIter, nIter]);
    nameSafe = regexprep(char(string(approachName)), '[^\w\-]', '_');
    stem = sprintf('%s_%dx%d_it', nameSafe, nelx, nely);
    itFmt = sprintf('%%0%dd', max(3, numel(sprintf('%d', nIter))));

    if exist(outDir, 'dir') ~= 7
        mkdir(outDir);
    end
    stale = dir(fullfile(outDir, [stem '*.png']));
    for s = 1:numel(stale)
        delete(fullfile(stale(s).folder, stale(s).name));
    end

    pngPaths = cell(1, numel(iters));
    for n = 1:numel(iters)
        k = iters(n);
        opts = struct('ApproachName', approachName, ...
            'StateLabel', sprintf('iteration %d', k), ...
            'Visible', 'off', 'Export', false, 'CloseFigure', false);
        if ~isempty(omegaHist)
            opts.Omega1 = omegaHist(k, 1);
            if size(omegaHist, 2) >= 2
                opts.Omega2 = omegaHist(k, 2);
            end
        end
        info = renderTopologyDensity(xHist(:, k), nelx, nely, opts);
        pngPaths{n} = fullfile(outDir, [stem sprintf(itFmt, k) '.png']);
        exportgraphics(info.figure, pngPaths{n}, 'Resolution', 160, 'BackgroundColor', 'white');
        close(info.figure);
    end
    fprintf('Saved %d topology snapshots (every %d iterations + last) in %s\n', ...
        numel(iters), saveEvery, outDir);
end
