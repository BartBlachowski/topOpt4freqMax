function stopstudy_olhoff(outDir, nelx, nely, maxOuter)
repo = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','Olhoff'));
addpath(fullfile(repo,'tools','Matlab'));
maxNumCompThreads(1);
pathScrub = olhoffcurrent_scrub_forbidden_paths(repo); %#ok<NASGU>
guard = olhoffcurrent_paths(); %#ok<NASGU>
preset = olhoffcurrent_production_preset().name;
cfg = olhoffcurrent_config(nelx, nely, 'Preset', preset, 'Diagnostics', true, ...
    'MaxOuter', maxOuter, 'StopRelativeChangeTolerance', 1e-12);   % stop effectively disabled
cfgProd = olhoffcurrent_config(nelx, nely, 'Preset', preset);       % production stop settings, for replay
stopProd = cfgProd.stop; moveProd = cfgProd.move;
t = tic; res = olhoffSolve(cfg); wall = toc(t);
hist = res.hist;
nOuter = numel(hist.dxOuter);
assert(numel(res.diag.drho) == nOuter);
rho = cfg.design.initial*ones(numel(res.rho),1);
X = zeros(numel(rho), nOuter);
for k = 1:nOuter
    rho = min(1, max(cfg.design.minimum, rho + res.diag.drho{k}));
    X(:,k) = rho;
end
assert(isequal(rho, res.rho), 'replay mismatch');
x0 = cfg.design.initial*ones(numel(rho),1);
M = stopstudy_metrics(X, x0);
M.dxOuter = hist.dxOuter(:); M.dxNorm2 = hist.dxNorm2(:); M.move = hist.move(:);
M.omega1 = hist.omega(1,:).'; M.omega2 = hist.omega(2,:).'; M.omega3 = hist.omega(3,:).';
M.nInner = hist.nInner(:); M.tOuter = hist.tOuter(:);
if isfield(hist,'stage'), M.stage = hist.stage(:); end
dxRel = []; if isfield(res,'aux') && isfield(res.aux,'dxRel'), dxRel = res.aux.dxRel(:); end
status = res.status;
every = 25; snapIdx = unique([every:every:nOuter, nOuter]);
Xsnap = X(:, snapIdx);
save(fullfile(outDir, sprintf('olhoff_%dx%d.mat', nelx, nely)), ...
    'M','dxRel','nOuter','wall','status','stopProd','moveProd','preset','nelx','nely','snapIdx','Xsnap','-v7.3');
T = struct2table(M);
writetable(T, fullfile(outDir, sprintf('olhoff_%dx%d.csv', nelx, nely)));
fprintf('DONE olhoff %dx%d: %d outer, status %s, wall %.1f s, w1 %.3f w2 %.3f Mnd %.4f\n', ...
    nelx, nely, nOuter, status, wall, M.omega1(end), M.omega2(end), M.Mnd(end));
end
