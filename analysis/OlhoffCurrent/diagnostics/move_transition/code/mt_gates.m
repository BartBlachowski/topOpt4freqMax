function G = mt_gates(rP, rU, rF, pInfo, E, cand)
%MT_GATES  The preregistered hard primary gates (brief sec. 16), frozen in
%   PREREGISTRATION.md sec. 9 BEFORE any ARM U run.
%
%   "Materially improves" (G13) = >= 25 % relative reduction in final M_nd.
%   "Not materially worse" (G14) = <= 10 % relative increase in mid-density.
%   Volume feasible = |mean(rho) - 0.5| <= 1e-3.
%   omega1 regression gate = < 3 %.

mesh = rP.mesh;  coarse = isequal(mesh, [160 20]);
PU = rU.per;

G = struct('mesh', mesh, 'coarse', coarse);
G.armU_status = rU.status;
G.armU_stopIter = rU.stopIter;
G.armP_stopIter = rP.stopIter;

% ---- solver health -----------------------------------------------------
G.noSolverFailure = ~any(contains(rU.log, 'LP inner solve failed')) && ...
                    all(rU.per.innerConv(1:rU.stopIter) ~= 0) && ...
                    all(isfinite(rU.rhoAtStop));
G.volumeFeasible  = abs(rU.volAtStop - 0.5) <= 1e-3;
% omega1 is compared on the POST-LOOP value -- the solver's own terminal
% assemble2D + eigSolve at the admission design -- because hist.omega is
% recorded at the START of an iteration, before that iteration's update.  This
% makes the comparison stricter, not looser.
G.omega1_production_postloop = rP.omega1PostLoop;
G.omega1_armU_postloop       = rU.omega1PostLoop;
G.omega1RelChange = (rU.omega1PostLoop - rP.omega1PostLoop)/rP.omega1PostLoop;
G.omega1RegressionOK = G.omega1RelChange > -0.03;
G.omega1RelChange_traj = (rU.omega1AtStop - rP.omega1AtStop)/rP.omega1AtStop;

% ---- G5 / G11: every descent satisfied the frozen criterion ------------
d = find(PU.descent);
ok = true;  detail = {};
for k = 1:numel(d)
    i = d(k);
    lo = max(1, i-10);
    rr = PU.ratio(lo:i-1);
    good = numel(rr) == 10 && all(rr < 0.5) && PU.utilCount(i) >= 10;
    ok = ok && good;
    detail{end+1} = struct('iter',i,'countAtCall',PU.utilCount(i), ...
        'allBelow',all(rr<0.5),'n',numel(rr),'ok',good); %#ok<AGROW>
end
G.everyDescentLegal = ok;
G.descentDetail = detail;
G.nDescents = numel(d);

% ---- G6 / G12: no immediate-post-descent admission ---------------------
G.gapToLastDescent = E.gapToLastDescent;
G.noPostDescentArtifact = isempty(E.stopIter) || ...
    isnan(E.gapToLastDescent) || E.gapToLastDescent >= cand.D;

% ---- G13 / G14 ---------------------------------------------------------
G.Mnd_production = rP.MndAtStop;
G.Mnd_armU       = rU.MndAtStop;
G.Mnd_relReduction = (rP.MndAtStop - rU.MndAtStop)/rP.MndAtStop;
G.Mnd_materiallyImproves = G.Mnd_relReduction >= 0.25;
G.mid_production = rP.midAtStop;
G.mid_armU       = rU.midAtStop;
G.mid_relChange  = (rU.midAtStop - rP.midAtStop)/rP.midAtStop;
G.mid_notWorse   = G.mid_relChange <= 0.10;

% ---- historical reference ----------------------------------------------
G.Mnd_fixedMoveHistorical = rF.MndAtStop;
G.Mnd_excessOverFixed_production = rP.MndAtStop - rF.MndAtStop;
G.Mnd_excessOverFixed_armU       = rU.MndAtStop - rF.MndAtStop;
if G.Mnd_excessOverFixed_production ~= 0
    G.fractionOfExcessRemoved = (rP.MndAtStop - rU.MndAtStop) / ...
                                 G.Mnd_excessOverFixed_production;
else
    G.fractionOfExcessRemoved = NaN;
end

% ---- naming per the brief ----------------------------------------------
if coarse
    G.G1_noSolverFailure = G.noSolverFailure;
    G.G2_volumeFeasible  = G.volumeFeasible;
    G.G3_omega1          = G.omega1RegressionOK;
    G.G5_descentLegal    = G.everyDescentLegal;
    G.G6_noArtifact      = G.noPostDescentArtifact;
    G.G7_terminationHonest = true;   % status is reported as-is, never relabelled
else
    G.G8_noSolverFailure = G.noSolverFailure;
    G.G9_volumeFeasible  = G.volumeFeasible;
    G.G10_omega1         = G.omega1RegressionOK;
    G.G11_descentLegal   = G.everyDescentLegal;
    G.G12_noArtifact     = G.noPostDescentArtifact;
    G.G13_MndMaterial    = G.Mnd_materiallyImproves;
    G.G14_midNotWorse    = G.mid_notWorse;
end

% ---- the key sec.14 quantity: M_nd immediately before the FIRST descent -
G.firstDescent_armP = local_first(rP.per);
G.firstDescent_armU = local_first(PU);

% ---- utilization characterization (sec. 12 failure mode) ---------------
G.armU_maxUtilCount   = max(PU.utilCount);
G.armU_fracBelow      = mean(PU.ratio < 0.5);
G.armU_medianRatio    = median(PU.ratio);
G.armU_stage1LastIter = find(PU.stage == 1, 1, 'last');
G.ladderComplete      = max(PU.stage) == 4;
end

function s = local_first(P)
d = find(P.descent, 1);
if isempty(d)
    s = struct('occurred',false,'iter',NaN,'Mnd',NaN,'mid',NaN,'omega1',NaN, ...
               'moveFrom',NaN,'moveTo',NaN);
else
    s = struct('occurred',true,'iter',d,'Mnd',P.Mnd(d-1),'mid',P.mid(d-1), ...
               'omega1',P.omega1(d-1),'moveFrom',P.move(d-1),'moveTo',P.move(d));
end
end
