function out = ma4_buildcheck()
%MA4_BUILDCHECK  Does THIS MATLAB build reproduce the archived 160x20 baseline?
%
%   PROVENANCE CONTROL, NOT EVIDENCE RECOVERY.  Preregistration sec. 10.
%
%   Every prior OlhoffCurrent study recorded MATLAB 25.2.0.3042426 (R2025b
%   Update 1).  This machine now has 25.2.0.2998904.  Combining a 400x50 result
%   produced here with 160x20/320x40 numbers produced there is unsound unless
%   the two builds are shown to do identical arithmetic.
%
%   So: rerun the production 160x20 baseline (91 outer iterations, ~3 min) and
%   compare against the COMMITTED per-iteration record
%   move_stop/runs/baseline_160x20_iterations.csv.  Nothing here recreates lost
%   evidence -- that CSV already exists; this only asks whether the arithmetic
%   still matches.
%
%   The CSV was written by writetable (~15 significant digits), so equality is
%   asserted at 1e-12 relative, and the DISCRETE facts -- outer count, stop
%   iteration, descent iterations -- must match exactly.

here = fileparts(mfilename('fullpath'));
study = fileparts(here);
diag  = fileparts(study);
csv   = fullfile(diag, 'move_stop', 'runs', 'baseline_160x20_iterations.csv');
assert(exist(csv,'file')==2, 'ma4_buildcheck:NoArchive', 'missing %s', csv);

maxNumCompThreads(1);
[cfg, meta] = ma4_config('P', 160, 20);   %#ok<ASGLU>

% assert this really is production, field by field over the whole schema
prod = olhoffcurrent_config(160, 20, 'Diagnostics', true);
S = olh.config.schema(); bad = {};
for k = 1:size(S,1)
    p = S{k,1};
    if strcmp(p,'runtime.name'); continue; end
    if ~isequaln(olh.config.getPath(cfg,p), olh.config.getPath(prod,p))
        bad{end+1} = p; %#ok<AGROW>
    end
end
assert(isempty(bad),'ma4_buildcheck:NotProduction','differs in: %s', strjoin(bad,', '));

fprintf('[buildcheck] running production 160x20 on MATLAB %s ...\n', version);
t = tic; res = olhoffSolve(cfg); wall = toc(t);
h = res.hist; nO = numel(h.N);

T = readtable(csv);
out = struct('matlabNow', version, 'matlabArchive', '25.2.0.3042426 (R2025b) Update 1', ...
             'wall_s', wall, 'nOuter_now', nO, 'nOuter_archive', height(T));

% ---- discrete facts must match EXACTLY ---------------------------------
out.nOuterMatch = (nO == height(T));
dNow = find(diff(h.move(:)) < 0) + 1;
dArc = T.outer(logical(T.moveDescent));
out.descents_now = dNow(:).'; out.descents_archive = dArc(:).';
out.descentMatch = isequal(dNow(:), dArc(:));

% ---- continuous fields to 1e-12 relative -------------------------------
n = min(nO, height(T));
cmp = { 'omega1', h.omega(1,1:n).', T.omega1(1:n) ; ...
        'volume', h.vol(1:n),       T.volume(1:n) ; ...
        'l2',     h.dxNorm2(1:n),   T.l2(1:n)     ; ...
        'maxAbs', h.dxOuter(1:n),   T.maxAbs(1:n) ; ...
        'move',   h.move(1:n),      T.move(1:n)   ; ...
        'beta',   h.beta(1:n),      T.beta(1:n)   };
out.maxRelErr = struct(); worst = 0;
for i = 1:size(cmp,1)
    a = double(cmp{i,2}(:)); b = double(cmp{i,3}(:));
    e = max(abs(a-b) ./ max(abs(b), 1e-300));
    out.maxRelErr.(cmp{i,1}) = e; worst = max(worst, e);
end
out.worstRelErr = worst;
out.reproduces = out.nOuterMatch && out.descentMatch && (worst < 1e-12);

fprintf('[buildcheck] nOuter %d vs %d (match=%d), descents [%s] vs [%s] (match=%d)\n', ...
        nO, height(T), out.nOuterMatch, num2str(out.descents_now), ...
        num2str(out.descents_archive), out.descentMatch);
fn = fieldnames(out.maxRelErr);
for i=1:numel(fn); fprintf('    %-8s max rel err = %.3e\n', fn{i}, out.maxRelErr.(fn{i})); end
fprintf('[buildcheck] REPRODUCES = %d   (wall %.0f s)\n', out.reproduces, wall);
end
