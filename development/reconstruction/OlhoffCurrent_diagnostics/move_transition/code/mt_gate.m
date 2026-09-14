function G = mt_gate(outP, RHO_P, nelx, nely, diagRoot)
%MT_GATE  BASELINE REGRESSION GATE (preregistration sec. 8).
%
%   ARM P is run through the candidate solver copy with the candidate OFF and
%   must reproduce the archived production trajectory.  This is what validates
%   mt_olhoffSolveT: the copy is not asserted to be faithful, it is PROVED to
%   reproduce the archive bitwise.
%
%   B-1  per-iteration history, 12 fields, bitwise vs admission_rule unstopped
%   B-2  design at the archived production stop iteration, bitwise vs move_stop
%   B-3  post-loop omega1 at that design, bitwise vs the archived value
%   B-4  production move transitions reproduced exactly

mesh = sprintf('%dx%d', nelx, nely);
G = struct('mesh', mesh, 'B1', false, 'B2', false, 'B3', false, 'B4', false, ...
           'detail', struct());

% ---- B-1 ---------------------------------------------------------------
A = load(fullfile(diagRoot,'admission_rule','runs',sprintf('unstopped_%s.mat',mesh)), 'out');
a = A.out.per;  p = outP.per;
fields = {'omega1','omega2','gap12','volume','move','stage','beta','l2', ...
          'maxAbs','nInner','innerConv','multN'};
n = min(numel(a.outer), numel(p.outer));
bad = {};
for k = 1:numel(fields)
    f = fields{k};
    if ~isequaln(a.(f)(1:n), p.(f)(1:n)); bad{end+1} = f; end %#ok<AGROW>
end
G.B1 = isempty(bad) && numel(a.outer)==numel(p.outer);
G.detail.B1_mismatched = bad;
G.detail.B1_nCompared  = n;
G.detail.B1_nArchive   = numel(a.outer);
G.detail.B1_nCandidate = numel(p.outer);

% ---- B-2 / B-3 ---------------------------------------------------------
B = load(fullfile(diagRoot,'move_stop','runs',sprintf('baseline_%s.mat',mesh)), 'out');
stopIter = B.out.nOuter;
G.detail.productionStopIter = stopIter;
G.B2 = isequaln(RHO_P(:,stopIter), double(B.out.rhoFinal(:)));
G.detail.B2_maxAbsDiff = max(abs(RHO_P(:,stopIter) - double(B.out.rhoFinal(:))));

w1arch = B.out.omega(1);
G.detail.archivedOmega1 = w1arch;
% Recompute the solver's own terminal analysis at that design.
cfgP = olhoffcurrent_config(nelx, nely, 'MaxOuter', 600, 'Diagnostics', true);
flat = olh.config.toLegacy(cfgP);
mdl  = model2D(flat);
massCfg = olh.config.getPath(cfgP,'material.mass');
[K,M] = assemble2D(mdl, RHO_P(:,stopIter), olh.config.getPath(cfgP,'material.stiffness.p'), massCfg);
Jc = olh.config.getPath(cfgP,'eigen.targetMode') + olh.config.getPath(cfgP,'eigen.maxCluster');
w = eigSolve(K, M, Jc, olh.config.getPath(cfgP,'eigen.solver'));
G.detail.recomputedOmega1 = w(1);
G.B3 = isequaln(w(1), w1arch);

% ---- B-4 ---------------------------------------------------------------
expected = struct('x160x20',[79 90 101],'x320x40',[130 141 152]);
key = sprintf('x%dx%d', nelx, nely);
got = p.outer(p.descent).';
G.detail.transitionsExpected = expected.(key);
G.detail.transitionsObserved = got;
G.B4 = isequal(got, expected.(key));

G.ok = G.B1 && G.B2 && G.B3 && G.B4;
fprintf(['[mt_gate] %s  B1=%d(%d fields bad) B2=%d(max|d|=%.3g) ' ...
         'B3=%d(%.17g vs %.17g) B4=%d [%s]  => %s\n'], mesh, ...
        G.B1, numel(bad), G.B2, G.detail.B2_maxAbsDiff, G.B3, ...
        G.detail.recomputedOmega1, w1arch, G.B4, num2str(got), ...
        local_tf(G.ok));
end

function s = local_tf(t), if t, s='PASS'; else, s='FAIL'; end, end
