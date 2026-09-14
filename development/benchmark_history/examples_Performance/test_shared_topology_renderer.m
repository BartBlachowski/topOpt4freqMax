function test_shared_topology_renderer
%TEST_SHARED_TOPOLOGY_RENDERER Smoke/regression coverage for WP0A.
%
% Proves that final density fields from Proposed, Yuksel and Olhoff enter one
% renderer and receive identical output geometry/presentation. It also proves
% explicit failure/unavailable statuses cannot create final-result files and
% that the iteration-efficiency grid delegates its cells to the same helper.

fprintf('=== test_shared_topology_renderer ===\n');
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'analysis', 'iteration_efficiency_study_design'));

outDir = tempname;
mkdir(outDir);
guard = onCleanup(@() localCleanup(outDir));

nelx = 12;
nely = 3;
[xx, yy] = meshgrid(linspace(0,1,nelx), linspace(0,1,nely));
fields = { ...
    0.15 + 0.80*xx, ...                    % Olhoff-shaped input
    0.10 + 0.85*(1-yy).*xx, ...            % Yuksel-shaped input
    0.05 + 0.90*(0.5*xx + 0.5*yy)};        % Proposed-shaped input
methods = {'Olhoff','Yuksel','OurApproach'};
statuses = {'VALID_STABILIZED_STATE_AT_FIXED_WORK', ...
            'NATIVE_CONVERGED', 'NATIVE_CONVERGED'};

infos = cell(1,3);
for i = 1:3
    opts = struct( ...
        'ApproachName', methods{i}, ...
        'OutputDir', outDir, ...
        'VisualizationQuality', 'regular', ...
        'DomainExtent', [0 8 0 1], ...
        'FigurePosition', [100 100 1000 260], ...
        'Visible', 'off', ...
        'ResultStatus', statuses{i}, ...
        'Admissible', true, ...
        'StateKind', 'final', ...
        'OverlayPolicy', 'none');
    infos{i} = renderTopologyDensity(fields{i}, nelx, nely, opts);
    assert(~infos{i}.skipped, '%s final field was unexpectedly skipped.', methods{i});
    assert(exist(infos{i}.png_path, 'file') == 2, ...
        '%s PNG was not exported.', methods{i});
    assert(exist(infos{i}.fig_path, 'file') == 2, ...
        '%s FIG was not exported.', methods{i});
end

geometryFields = {'x_lim','y_lim','y_dir','clim','data_aspect_ratio', ...
    'figure_size_px','display_image_size','overlay_policy'};
for i = 2:3
    for k = 1:numel(geometryFields)
        field = geometryFields{k};
        assert(isequal(infos{1}.geometry.(field), infos{i}.geometry.(field)), ...
            'Geometry field %s differs between %s and %s.', ...
            field, methods{1}, methods{i});
    end
end
assert(isequal(infos{1}.geometry.x_lim, [0 8]));
assert(isequal(infos{1}.geometry.y_lim, [0 1]));
assert(isequal(infos{1}.geometry.clim, [0 1]));
assert(strcmp(infos{1}.geometry.y_dir, 'normal'));
assert(isequal(infos{1}.geometry.figure_size_px, [1000 260]));

blockedBase = fullfile(outDir, sprintf('Olhoff_%dx%d', nelx, nely));
blocked = renderTopologyDensity(fields{1}, nelx, nely, struct( ...
    'ApproachName', 'Olhoff', 'OutputDir', outDir, ...
    'OutputBase', 'must_not_exist_failure', ...
    'Visible', 'off', 'ResultStatus', 'SOLVER_FAILURE', ...
    'Admissible', false, 'StateKind', 'final'));
assert(blocked.skipped, 'SOLVER_FAILURE must be refused.');
assert(exist(fullfile(outDir, 'must_not_exist_failure.png'), 'file') ~= 2);

unavailable = renderTopologyDensity([], nelx, nely, struct( ...
    'ApproachName', 'Olhoff', 'OutputDir', outDir, ...
    'OutputBase', 'must_not_exist_unavailable', ...
    'Visible', 'off', 'ResultStatus', 'RUN_ERROR', ...
    'Admissible', false, 'StateKind', 'final'));
assert(unavailable.skipped, 'RUN_ERROR must be refused.');
assert(exist(fullfile(outDir, 'must_not_exist_unavailable.png'), 'file') ~= 2);

% The accepted-state grid is deliberately a selector/layout layer. Its cell
% records identify the public shared renderer, and unavailable cells stay empty.
template = struct('density', [], 'nelx', nelx, 'nely', nely, 'method', '', ...
    'status', '', 'admissible', false, 'state_label', '', ...
    'domain_extent', [0 8 0 1], 'representation', 'raw', ...
    'support_points', [0 0.5; 8 0.5], 'load_vectors', zeros(0,4));
states = repmat(template, 1, 4);
for i = 1:3
    states(i).density = fields{i};
    states(i).method = methods{i};
    states(i).status = statuses{i};
    states(i).admissible = true;
    states(i).state_label = 'k_enter';
end
states(4).method = 'Olhoff';
states(4).status = 'UNAVAILABLE';
states(4).state_label = 'unavailable';

gridBase = fullfile(outDir, 'accepted_state_grid_smoke');
grid = render_iteration_efficiency_topology_grid(states, gridBase, ...
    struct('GridSize', [2 2], 'Visible', 'off', 'OverlayPolicy', 'supports'));
assert(strcmp(grid.renderer, 'tools/Matlab/renderTopologyDensity.m'));
assert(all(cellfun(@(c) ~c.skipped, grid.cells(1:3))));
assert(grid.cells{4}.skipped, 'Unavailable grid cell must not contain a topology.');
assert(exist(grid.png_path, 'file') == 2 && exist(grid.pdf_path, 'file') == 2 && ...
    exist(grid.fig_path, 'file') == 2, 'Grid exports were not produced.');

% Suppress an unused-variable warning while retaining a readable expected
% naming-convention expression next to the assertions above.
assert(exist([blockedBase '.png'], 'file') == 2);
fprintf('PASS: shared renderer geometry, naming, status gate, and grid reuse.\n');
end

function localCleanup(path)
if exist(path, 'dir') == 7
    rmdir(path, 's');
end
close all force;
end
