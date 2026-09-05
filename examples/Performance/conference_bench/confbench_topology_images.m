function info = confbench_topology_images(cfg, records)
%CONFBENCH_TOPOLOGY_IMAGES  Final topology image for every campaign run.
%
%   info = CONFBENCH_TOPOLOGY_IMAGES(cfg, records) renders the final density
%   field of each record into
%
%       <cfg.outputDir>/topologies/topology_<methodkey>_<nelx>x<nely>.png (+ .fig)
%
%   so a 9-mesh, 3-method campaign yields 27 images.  Nothing is recomputed:
%   the design vector records(i).x is the state the solver finished on and was
%   saved with the campaign, so these images are a view of the recorded result,
%   not a second solve.
%
%   ONE RENDERER.  Every image goes through tools/Matlab/renderTopologyDensity,
%   the shared status-aware renderer fixed by WP0A_VISUALIZATION_FIX.md.  That
%   is what makes topologies from three architecturally different methods
%   visually comparable: identical orientation, identical [0,1] density scale,
%   identical domain extent and aspect.  A private imagesc here would silently
%   reintroduce the divergence WP0A removed.
%
%   WHAT IS NOT DRAWN.  The renderer refuses a final image for a failed,
%   errored, unavailable or not-run record, and this function refuses one for a
%   record with no usable design vector.  Refusals are counted and listed, so a
%   missing image is a reported fact rather than a gap in a directory.
%
%   A run that finished on its iteration cap (ok == false, e.g. CAP_HIT) DID
%   produce a real final design and is rendered -- but its title carries the
%   status, because a budget-truncated design must not be read as a converged
%   one.
%
%   Produced OUTSIDE every solver timing boundary, from recorded results only.
%
%   See also RENDERTOPOLOGYDENSITY, CONFBENCH_COMPLEXITY_PLOTS.

info = struct('dir', '', 'png_paths', {{}}, 'n_written', 0, 'n_skipped', 0, ...
    'skipped', {{}});

outDir = fullfile(cfg.outputDir, 'topologies');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end
info.dir = outDir;

% The benchmark domain is one shared 8 x 1 beam for all three methods
% (study_base_config: domain.size 8.0 x 1.0; olhoffm4_config: a = 8, b = 1).
% Passing it explicitly is what pins every image to the same extent and aspect
% instead of the renderer's legacy element-index framing, which would make a
% 160x20 and an 800x100 design impossible to compare side by side.
domainExtent  = [0 8 0 1];
figurePosition = [100 100 1000 260];

fprintf('---------------- FINAL TOPOLOGIES ----------------\n');
for i = 1:numel(records)
    r = records(i);
    if isfield(r, 'is_warmup') && r.is_warmup
        continue
    end

    nelx = r.mesh(1);
    nely = r.mesh(2);
    tag  = sprintf('%-30s %4dx%-4d', r.method, nelx, nely);

    % A record can be missing its design for an ordinary reason (the run
    % errored).  Decide that here rather than letting the renderer fail.
    hasField = isfield(r, 'x') && ~isempty(r.x) && numel(r.x) == nelx*nely ...
        && all(isfinite(r.x(:)));
    errored  = isfield(r, 'error') && ~isempty(r.error);
    if ~hasField || errored
        why = 'no usable final density field was recorded';
        if errored; why = sprintf('the run errored (%s)', r.error); end
        fprintf('  %s  SKIPPED: %s\n', tag, why);
        info.n_skipped = info.n_skipped + 1;
        info.skipped{end+1} = sprintf('%s %dx%d: %s', r.method, nelx, nely, why);
        continue
    end

    % Frequencies for the title, when the record carries them.
    om = NaN(1,2);
    if isfield(r, 'omega') && numel(r.omega) >= 1
        om(1) = r.omega(1);
        if numel(r.omega) >= 2; om(2) = r.omega(2); end
    elseif isfield(r, 'omega1_native')
        om(1) = r.omega1_native;
    end

    % A run that stopped on its budget rather than on its own convergence test
    % is labelled on the image itself.
    stateLabel = '';
    if isfield(r, 'ok') && ~r.ok
        % The renderer sets the title with Interpreter = 'tex', where a bare
        % underscore makes the next character a subscript: CAP_HIT would be
        % drawn as CAP with a subscript H.  Escaping keeps the status token
        % identical to the one in the CSV, which is what makes an image
        % traceable back to its row.
        stateLabel = sprintf('%s -- NOT NATIVELY CONVERGED', ...
            strrep(strtrim(r.status), '_', '\_'));
    end

    opts = struct( ...
        'ApproachName', r.method, ...
        'OutputDir', outDir, ...
        'OutputBase', sprintf('topology_%s_%dx%d', r.method_key, nelx, nely), ...
        'Omega1', om(1), 'Omega2', om(2), ...
        'VisualizationQuality', 'regular', ...
        'DomainExtent', domainExtent, ...
        'FigurePosition', figurePosition, ...
        'Visible', 'off', ...
        'ResultStatus', r.status, ...
        'Admissible', true, ...
        'StateKind', 'final', ...
        'StateLabel', stateLabel, ...
        'OverlayPolicy', 'none', ...
        'Export', true, ...
        'CloseFigure', true);

    try
        ri = renderTopologyDensity(r.x, nelx, nely, opts);
    catch renderErr
        fprintf('  %s  SKIPPED: renderer failed (%s)\n', tag, renderErr.message);
        info.n_skipped = info.n_skipped + 1;
        info.skipped{end+1} = sprintf('%s %dx%d: renderer failed (%s)', ...
            r.method, nelx, nely, renderErr.message);
        continue
    end

    if ri.skipped
        fprintf('  %s  SKIPPED: %s\n', tag, ri.skip_reason);
        info.n_skipped = info.n_skipped + 1;
        info.skipped{end+1} = sprintf('%s %dx%d: %s', r.method, nelx, nely, ri.skip_reason);
        continue
    end

    info.n_written = info.n_written + 1;
    info.png_paths{end+1} = ri.png_path;
end

fprintf('  %d image(s) written to %s\n', info.n_written, outDir);
if info.n_skipped > 0
    fprintf('  %d record(s) produced NO image:\n', info.n_skipped);
    for i = 1:numel(info.skipped)
        fprintf('      %s\n', info.skipped{i});
    end
end
fprintf('\n');
end
