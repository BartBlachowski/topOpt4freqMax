function P = cv_telemetry(res, RHO, NE, rho0)
%CV_TELEMETRY  Full per-iteration telemetry for one candidate run (Phase 12).
%
%   Built from the FROZEN definitions by direct reuse of
%   dynamical_regime/scripts/dr_telemetry.m -- called, never re-typed -- plus:
%
%     * the controller trace the solver actually acted on (hist.ex*), which is
%       STAGE-LOCAL: at the start of each move stage its windows restart, so it
%       differs from the global dynamical quantities by construction;
%     * the production counterfactual: what the beta-stall detector and the
%       sec. 3.5.1 design-change stop WOULD have done on this same trajectory.
%
%   The counterfactual is a PREDICATE REPLAY on the realized path, not a
%   counterfactual trajectory: production under its own rule would have taken a
%   different path after its first descent.  Stated so it is not over-read.

P = dr_telemetry(res.hist, RHO, NE, rho0);       % frozen definitions, by reference
h = res.hist;
n = numel(h.N);

P.cumInner = h.cumInner(:);
P.tOuter   = h.tOuter(:);
P.dxOuter  = h.dxOuter(:);
P.volErr   = h.volErr(:);

% ---- the controller trace, exactly as the solver read it ----------------
f = {'exA','exB','exE','exNA','exNB','exDecl','exCos','exNet','exMedcos', ...
     'exMednet','exAmp','exStageStart'};
for i = 1:numel(f)
    v = h.(f{i});
    if isempty(v), v = nan(n,1); end
    P.(f{i}) = v(:);
end
P.exTol = res.exhaustion.tol*ones(n,1);

% ---- production counterfactual: the sec. 3.5.1 stop and its guard --------
tol = olh.config.getPath(res.cfg,'stop.tolerance');
P.prodTol       = tol*ones(n,1);
P.prodStopRaw   = P.l2 < tol;                                  % ||drho||_2 < eps
P.prodSettled   = [false; P.move(2:end) == P.move(1:end-1)];
P.prodStopAdmit = P.prodStopRaw & P.prodSettled;               % what production would admit
% the inherited NATIVE stop, first iteration it holds on a settled move
P.nativeStopIter = NaN;
k = find(P.prodStopAdmit, 1);
if ~isempty(k), P.nativeStopIter = k; end

% ---- production counterfactual: the beta ladder, replayed ---------------
% P.betaStallRel / P.betaStallFires come from dr_telemetry (W=10, tol=5e-3,
% beta history through k-1) and are character-identical to olh.move.limit's
% boundVariable branch.  Here we additionally replay the STAGE production would
% have reached, including its re-arm dwell.
W = olh.config.getPath(res.cfg,'move.continuation.window');
sTol = olh.config.getPath(res.cfg,'move.continuation.tolerance');
levels = olh.config.getPath(res.cfg,'move.levels');
stage = 1; lastStage = 0;
P.prodStageShadow = ones(n,1); P.prodMoveShadow = levels(1)*ones(n,1);
for kk = 1:n
    b = P.beta(1:kk-1);
    if numel(b) >= 2*W && (kk - lastStage) > W
        w2 = mean(b(end-W+1:end)); w1 = mean(b(end-2*W+1:end-W));
        if (w2-w1)/max(abs(w1),eps) < sTol
            stage = min(stage+1, numel(levels)); lastStage = kk;
        end
    end
    P.prodStageShadow(kk) = stage; P.prodMoveShadow(kk) = levels(stage);
end
end
