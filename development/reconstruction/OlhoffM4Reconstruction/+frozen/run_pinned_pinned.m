function res = run_pinned_pinned(varargin)
%RUN_PINNED_PINNED  Editable pinned-pinned Du-Olhoff reconstruction (M4).
%
%   THIS FILE SITS IN A NAMESPACE FOLDER (+frozen), so the Editor's Run button
%   cannot reach it from a cold start: MATLAB refuses to put a +folder on the
%   path, and no name resolves from inside one. Make the PARENT folder the
%   current folder, or put it on the path, and call the QUALIFIED name:
%
%     addpath('<repo>/analysis/OlhoffM4Reconstruction');
%     res = frozen.run_pinned_pinned();
%     res = frozen.run_pinned_pinned('nelx', 320, 'nely', 40, ...
%         'rminEl', 2.4, 'tolInner', 0.02, 'maxOuter', 600);
%
%   Edit USER PARAMETERS below to change the defaults in place; every one of
%   them also accepts a name/value override on the call above. Defaults are
%   the current conference configuration at its first mesh, 160x20. Both ux
%   and uy are fixed at the mid-height node of EACH end; nely must be even.
%   The domain is 8x1, with 50% volume, Q4 elements and consistent mass.
%
%   rminEl is the filter radius in ELEMENT units and is passed to prepFilter
%   unchanged; the physical radius it corresponds to is rminEl*(b/nely). Note
%   the consequence: holding rminEl fixed while refining the mesh SHRINKS the
%   filter physically, so a mesh study run this way is not filter-controlled.
%   The frozen conference filter is R = 0.06 physical, i.e. rminEl = 0.06*nely
%   -- 1.2 at 160x20, the default here. rminEl <= 1 reaches no neighbour and
%   leaves the sensitivities unfiltered.
%
%   tolOuterRef is the L2 density-change tolerance at 3200 elements; it scales
%   by sqrt(nelx*nely/3200), preserving the RMS tolerance across meshes.
%   tolInner tests the relative infinity-norm change of the inner density
%   increment after at least minInner MMA iterations.
%
%   res contains the effective cfg, rho, omega (rad/s), hist, log, model and
%   stopping status. hist.omega records pre-update frequencies; res.omega is
%   recomputed at the final design. Save results with save('run.mat','res').
%   Edited settings describe a custom M4 run, not the frozen benchmark.
%
%   See also OLHOFFM4_CONFIG, OLHOFFM4_PATHS.

%% USER PARAMETERS -- current benchmark defaults
params.nelx = 800;
params.nely = 100;
params.rminEl = 1.2;              % filter radius in ELEMENT units; = 0.06 physical at 160x20

params.tolOuterRef = 0.05;        % L2 at 3200 elements; RMS = 8.838835e-4
params.tolInner = 0.05;           % relative change in inner density increment
params.minInner = 5;             % minimum MMA iterations per outer step
params.maxInner = 500;           % MMA safety budget per outer step
params.maxOuter = 400;           % outer safety budget; reaching it is a cap

params.s2Levels = [0.04 0.02 0.01 0.005]; % descending move-limit ladder
params.s2Window = 10;            % window length for the beta stall detector
params.s2Tol = 5e-3;             % relative change between beta window means

params.verbose = true;          % set true to print each outer iteration
params.showPlots = true;         % final topology and first three frequencies

%% Parse and validate before constructing the model
p = inputParser();
p.FunctionName = 'frozen.run_pinned_pinned';
p.PartialMatching = false;
names = fieldnames(params);
for i = 1:numel(names)
    p.addParameter(names{i}, params.(names{i}));
end
p.parse(varargin{:});
opts = p.Results;

integerFields = {'nelx', 'nely', 'minInner', 'maxInner', 'maxOuter', 's2Window'};
for i = 1:numel(integerFields)
    key = integerFields{i};
    validateattributes(opts.(key), {'numeric'}, ...
        {'real', 'scalar', 'finite', 'integer', 'positive'}, p.FunctionName, key);
end
positiveFields = {'rminEl', 'tolOuterRef', 'tolInner', 's2Tol'};
for i = 1:numel(positiveFields)
    key = positiveFields{i};
    validateattributes(opts.(key), {'numeric'}, ...
        {'real', 'scalar', 'finite', 'positive'}, p.FunctionName, key);
end
validateattributes(opts.s2Levels, {'numeric'}, ...
    {'real', 'vector', 'nonempty', 'finite', 'positive', '<=', 1}, ...
    p.FunctionName, 's2Levels');
assert(all(diff(opts.s2Levels) < 0), 'run_pinned_pinned:MoveLadder', ...
    's2Levels must be strictly decreasing.');
assert(opts.minInner <= opts.maxInner, 'run_pinned_pinned:InnerBudget', ...
    'minInner must not exceed maxInner.');
validateattributes(opts.verbose, {'logical'}, {'scalar'}, p.FunctionName, 'verbose');
validateattributes(opts.showPlots, {'logical'}, {'scalar'}, p.FunctionName, 'showPlots');

%% Start from the conference configuration and apply only requested controls
root = fileparts(fileparts(mfilename('fullpath')));
entryPath = path();
pathCleanup = onCleanup(@() path(entryPath));
addpath(root);
cfg = olhoffm4_config(opts.nelx, opts.nely);
% Element units are set DIRECTLY. rminPhys must be left empty: olhoffOpt
% overwrites rminEl with rminPhys/(b/nely) whenever rminPhys is a positive
% scalar, which is the precedence defaultCfg.m records.
cfg.rminEl = double(opts.rminEl);
cfg.rminPhys = [];
cfg.tolOuter = double(opts.tolOuterRef)*sqrt(cfg.nelx*cfg.nely/3200);
cfg.tolInner = double(opts.tolInner);
cfg.minInner = double(opts.minInner);
cfg.maxInner = double(opts.maxInner);
cfg.maxOuter = double(opts.maxOuter);
cfg.s2Levels = double(opts.s2Levels(:).');
cfg.move = cfg.s2Levels(1);
cfg.s2Window = double(opts.s2Window);
cfg.s2Tol = double(opts.s2Tol);
cfg.verbose = opts.verbose;
% Inherited: bc='a', support='mid', axial='both', multRule='subspace',
% subN=2, published MMA, filterMode='all', outerGuard='settledmove',
% legacy beta continuation signal, diag=false, threads=1.

fprintf('\nDu-Olhoff reconstruction (M4): pinned-pinned beam\n');
fprintf('  Mesh: %dx%d; both end mid-height nodes fixed in ux and uy\n', cfg.nelx, cfg.nely);
fprintf('  Filter: rminEl=%.6g elements, R=%.6g physical\n', ...
    cfg.rminEl, cfg.rminEl*(cfg.b/cfg.nely));
fprintf('  Outer tolerance: L2=%.6g, RMS=%.6g (settled-move guard)\n', ...
    cfg.tolOuter, cfg.tolOuter/sqrt(cfg.nelx*cfg.nely));
fprintf('  Inner relative tolerance: %.6g; budgets: outer=%d, inner=%d..%d\n', ...
    cfg.tolInner, cfg.maxOuter, cfg.minInner, cfg.maxInner);

%% Run through the existing implementation guard
entryThreads = maxNumCompThreads();
threadCleanup = onCleanup(@() maxNumCompThreads(entryThreads));
res = solveGuarded(cfg);
res.method_label = 'Du-Olhoff reconstruction (M4)';
res.converged = any(contains(res.log, 'converged at outer iteration'));
if res.converged
    res.status = 'NATIVE_CONVERGED';
else
    res.status = 'CAP_HIT';
end
fprintf('  %s: %d outer, %d total inner iterations, %.3f s\n', ...
    res.status, res.nOuter, sum(res.hist.nInner), res.wallclock);
fprintf('  Final omega1=%.6g rad/s; mean density=%.6g; unconverged inner solves=%d\n', ...
    res.omega(1), mean(res.rho), sum(~res.hist.innerConv));

%% Final design and frequency history
if opts.showPlots
    figure('Name', 'Du-Olhoff reconstruction (M4): pinned-pinned', 'Color', 'w');
    tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax = nexttile;
    imagesc(ax, [res.mdl.dx/2 cfg.a-res.mdl.dx/2], ...
        [res.mdl.dy/2 cfg.b-res.mdl.dy/2], reshape(res.rho, cfg.nely, cfg.nelx));
    axis(ax, 'equal');
    axis(ax, 'tight');
    colormap(ax, flipud(gray(256))); % solid black, void white
    clim(ax, [0 1]);
    xlabel(ax, 'x'); ylabel(ax, 'y (downward from top)');
    title(ax, sprintf('%s: \\omega_1 = %.4f rad/s', res.status, res.omega(1)), ...
        'Interpreter', 'tex');

    ax = nexttile;
    omegaTrace = [res.hist.omega(1:3,:), res.omega(1:3)];
    plot(ax, 0:res.nOuter, omegaTrace.', 'LineWidth', 1.2);
    grid(ax, 'on');
    xlabel(ax, 'Completed outer iterations');
    ylabel(ax, '\omega (rad/s)');
    legend(ax, {'\omega_1', '\omega_2', '\omega_3'}, 'Location', 'best');
end
end

function res = solveGuarded(cfg)
guard = olhoffm4_paths(); %#ok<NASGU>
res = olhoffOpt(cfg);
end
