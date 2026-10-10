function run_olhoff_ref(nelx, nely, initRhoFile, tag)
% Reference Du-Olhoff production run (and warm-started corrector runs) for the NSO study.
%   run_olhoff_ref(160, 20)                       cold start, saves results/olhoff_ref_160x20.mat
%   run_olhoff_ref(160, 20, 'rho.mat', 'pc165')   warm start from the density in rho.mat (variable rho),
%                                                 saves results/olhoff_corrector_160x20_pc165.mat
if nargin < 1, nelx = 160; end
if nargin < 2, nely = 20;  end
if nargin < 3, initRhoFile = ''; end
if nargin < 4, tag = ''; end
repo = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
outdir = fullfile(repo, 'examples', 'sand', 'olhoff_nso', 'results');
addpath(fullfile(repo, 'analysis', 'Olhoff'));
guard = olhoffcurrent_paths(); %#ok<NASGU>
pname = olhoffcurrent_production_preset().name;
fprintf('production preset: %s\n', pname);
[cfg, info] = olhoffcurrent_config(nelx, nely, 'Preset', pname); %#ok<ASGLU>
if isempty(initRhoFile)
    stem = sprintf('olhoff_ref_%dx%d', nelx, nely);
else
    stem = sprintf('olhoff_corrector_%dx%d_%s', nelx, nely, tag);
    s = load(initRhoFile);
    rho0 = double(s.rho(:));
    assert(numel(rho0) == nelx*nely, 'initial density has %d entries, mesh has %d', numel(rho0), nelx*nely);
    cfg = olh.config.setPath(cfg, 'design.initial', rho0);   % vector initial design (see olhoffSolve)
end
fid = fopen(fullfile(outdir, [stem '_cfg.json']), 'w');
try, fprintf(fid, '%s', jsonencode(cfg, 'PrettyPrint', true)); catch, end; fclose(fid);
try
    d = olh.config.describe(cfg);
    fid = fopen(fullfile(outdir, [stem '_describe.txt']), 'w');
    if ischar(d) || isstring(d); fprintf(fid, '%s', char(d)); else; fprintf(fid, '%s', jsonencode(d)); end
    fclose(fid);
catch ME
    fprintf('describe failed: %s\n', ME.message);
end
tCall = tic;
if isempty(initRhoFile)
    res = olhoffSolve(cfg);          % production solver, untouched
else
    res = olhoffSolveWarm(cfg);      % sandbox copy: one line changed to accept a vector initial design
end
wall = toc(tCall);
h = res.hist;
ref = struct();
ref.preset = pname; ref.nelx = nelx; ref.nely = nely; ref.init_file = initRhoFile; ref.tag = tag;
ref.rho = double(res.rho(:));
ref.omega = double(res.omega(:));
ref.lambda = double(res.lambda(:));
ref.status = res.status;
ref.wall_s = wall;
ref.nOuter = numel(h.N);
ref.hist_omega = double(h.omega);
ref.hist_N = double(h.N(:));
ref.hist_vol = double(h.vol(:));
ref.hist_dxNorm2 = double(h.dxNorm2(:));
ref.hist_dxOuter = double(h.dxOuter(:));
ref.hist_move = double(h.move(:));
ref.hist_tOuter = double(h.tOuter(:));
ref.hist_tEig = double(h.tEig(:));
ref.hist_tGrad = double(h.tGrad(:));
ref.hist_tInner = double(h.tInner(:));
ref.hist_nInner = double(h.nInner(:));
ref.hist_beta = double(h.beta(:));
ref.log = char(strjoin(string(res.log), newline));
ref.Mnd = mean(4*ref.rho.*(1-ref.rho));
save(fullfile(outdir, [stem '.mat']), '-struct', 'ref', '-v7');
fprintf('DONE %s status=%s nOuter=%d omega1=%.4f omega2=%.4f Mnd=%.4f wall=%.1fs\n', ...
    stem, res.status, ref.nOuter, ref.omega(1), ref.omega(2), ref.Mnd, wall);
end
