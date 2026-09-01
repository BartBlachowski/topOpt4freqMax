function run_olhoff_regularized_tests()
%RUN_OLHOFF_REGULARIZED_TESTS Route, accounting, controller and certificate tests.
%
%   Every case below is a SOFTWARE-MECHANICS test on a toy mesh: it checks
%   parsing, dispatch, bookkeeping, controller forcing and the stopping
%   classification.  None of them is evidence about convergence quality,
%   topology or mesh robustness -- see analysis/OlhoffRegularized/AUDIT_REPORT.md
%   for the >= 160x20 scientific runs.

nPass=0;
%% 1. the four routes execute, keep the volume, and respect their caps -------
routes={"olhoff","lp";"olhoff","mma";"ks","lp";"ks","mma"};
for i=1:size(routes,1)
    formulation=routes{i,1};optimizer=routes{i,2};
    rc=struct('verbose',false,'formulation',formulation,'optimizer',optimizer, ...
        'max_outer_iterations',2,'max_inner_iterations',150,'min_inner',5, ...
        'max_trial_steps',3,'persistence',2);
    [rho,w,info]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'fixedPinned',rc);
    assert(all(isfinite(rho))&&all(isfinite(w)),'Nonfinite route result.');
    assert(abs(mean(rho)-.5)<1e-4,'Volume constraint drifted.');
    assert(info.cfg.maxOuter==2&&info.cfg.maxInner==150,'Iteration limits were not authoritative.');
    assert(info.iterations.outer<=2&&info.iterations.trial_total<=6,'Iteration accounting exceeded its cap.');
    assert(strcmp(info.formulation,formulation)&&strcmp(info.optimizer,optimizer),'Route metadata mismatch.');
    assert(info.iterations.accepted_updates>=1,'Route %s/%s accepted no smoke update.',formulation,optimizer);
    assert(all(info.history.trustUsed<=info.history.moveCeilingUsed*(1+1e-12)), ...
        'Trust radius exceeded the persistent move ceiling.');
    assert(all(diff(info.history.moveCeilingNext)<=0),'Move ceiling increased.');
    assert(all(info.history.innerIterations<=info.cfg.maxInner*3),'Inner accounting exceeded its cap.');
    %   The full Eq. (25d) sub-eigenvalue coupling must be live for, and only
    %   for, the genuine Olhoff nested-MMA route.
    wantOffDiag=strcmp(formulation,'olhoff')&&strcmp(optimizer,'mma');
    assert(info.cfg.offDiag==wantOffDiag, ...
        'Route %s/%s ran with offDiag=%d.',formulation,optimizer,info.cfg.offDiag);
end
%   ... and the Eq. (16)/(22) LP route must refuse the coupled model outright.
try
    topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply', ...
        struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
               'max_outer_iterations',1,'off_diag',true));
catch ME
    assert(strcmp(ME.identifier,'topopt_olhoff_regularized:LpOffDiag')|| ...
           contains(ME.message,'offDiag'),'Unexpected LP/offDiag error: %s',ME.message);
end
nPass=nPass+1;

%% 2. forced stage exhaustion contracts the persistent ceiling ---------------
warnState=warning('off','topopt_olhoff_regularized:LooseStationarityTolerance');
cleanupWarn=onCleanup(@()warning(warnState)); %#ok<NASGU>
rcForce=struct('verbose',false,'formulation','olhoff','optimizer','mma', ...
    'max_outer_iterations',2,'max_inner_iterations',150,'min_inner',5, ...
    'max_trial_steps',3,'objective_tol',1e9,'progress_tolerance',1, ...
    'progress_spike_tolerance',1,'progress_window',1,'progress_dwell',1, ...
    'progress_shrink_factor',.5);
[~,~,info]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'fixedPinned',rcForce);
assert(info.iterations.move_ceiling_contractions>=1,'Move-ceiling controller did not contract.');
assert(any(info.history.moveCeilingContracted),'Move-ceiling contraction was not logged.');
assert(all(diff(info.history.moveCeilingNext)<=0),'Persistent move ceiling regrew after contraction.');
assert(all(info.history.trustNext<=info.history.moveCeilingNext*(1+1e-12)), ...
    'Post-update trust exceeded the persistent move ceiling.');
nPass=nPass+1;

%% 3. a significant single-step improvement blocks ceiling contraction -------
%   The cumulative-progress window alone is not enough: with progress_tolerance
%   wide open the cumulative test always passes, so any contraction that still
%   occurs must have been permitted by the spike guard.  Calibrate the guard
%   from the run's own observed maximum single-step progress.
rcSpike=rcForce;rcSpike.progress_window=3;rcSpike.progress_dwell=3;
rcSpike.max_outer_iterations=6;rcSpike.progress_tolerance=1e9;
rcSpike.progress_spike_tolerance=0;
[~,~,infoBlocked]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'fixedPinned',rcSpike);
observedSpike=max(infoBlocked.history.windowMaxStepProgress(~isnan(infoBlocked.history.windowMaxStepProgress)));
assert(~isempty(observedSpike)&&observedSpike>0,'No single-step progress was observed to guard against.');
assert(infoBlocked.iterations.move_ceiling_contractions==0, ...
    'A significant single-step improvement did not block move-ceiling contraction.');
assert(all(infoBlocked.history.windowCumulativeProgress(infoBlocked.history.progressWindowReady==1) ...
    <=rcSpike.progress_window*rcSpike.progress_tolerance), ...
    'The cumulative-progress test was not the passing one in the blocked case.');
rcSpike.progress_spike_tolerance=observedSpike*10;
[~,~,infoAllowed]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'fixedPinned',rcSpike);
assert(infoAllowed.iterations.move_ceiling_contractions>=1, ...
    'Relaxing only the spike guard did not permit contraction: the guard is not load-bearing.');
nPass=nPass+1;

%% 4. CAP_HIT can never satisfy the native-convergence classification --------
rcCap=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
    'max_outer_iterations',4,'max_trial_steps',3,'persistence',20);
[~,~,infoCap]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply',rcCap);
assert(strcmp(infoCap.status,'CAP_HIT'),'Expected CAP_HIT, got %s.',infoCap.status);
assert(strcmp(infoCap.stop_reason,'maximum_outer_iterations'),'CAP_HIT carried the wrong stop reason.');
assert(~strcmp(infoCap.status,'CONVERGED'),'CAP_HIT was classified as convergence.');
assert(all(infoCap.history.convergenceCount<infoCap.regularization.persistence), ...
    'A CAP_HIT run reached the convergence persistence without being classified CONVERGED.');
assert(infoCap.iterations.outer==infoCap.cfg.maxOuter,'CAP_HIT did not run to its cap.');
nPass=nPass+1;

%% 5. minimum trust WITHOUT stationarity is GLOBALIZATION_STALLED ------------
%   min_inner = max_inner with tol_inner = 0 makes every inner MMA solve report
%   non-convergence, so every trial is rejected and no stationarity measure is
%   ever produced (predSlope stays +Inf).  move_min = move puts the trust radius
%   at its floor from the first trial.
rcStall=struct('verbose',false,'formulation','olhoff','optimizer','mma', ...
    'max_outer_iterations',3,'max_inner_iterations',4,'min_inner',4, ...
    'tol_inner',0,'max_trial_steps',2,'move_min',.005,'move_max',.005);
[~,~,infoStall]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'fixedPinned',rcStall);
assert(strcmp(infoStall.status,'GLOBALIZATION_STALLED'), ...
    'Expected GLOBALIZATION_STALLED, got %s.',infoStall.status);
assert(strcmp(infoStall.stop_reason,'minimum_trust_radius_without_stationarity'), ...
    'Stall carried the wrong stop reason.');
assert(infoStall.iterations.accepted_updates==0,'A stalled run accepted an update.');
assert(~any(infoStall.history.innerConverged),'A stalled run reported a converged inner solve.');
nPass=nPass+1;

%% 6. convergence requires the DECLARED persistence --------------------------
%   Same forced-stationarity configuration, two persistence values.
rcConv=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
    'max_outer_iterations',40,'max_trial_steps',3, ...
    'objective_tol',1e9,'density_tol',1,'rms_density_tol',1,'persistence',3);
[~,~,infoP3]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply',rcConv);
assert(strcmp(infoP3.status,'CONVERGED'),'Forced-stationarity case did not converge.');
assert(infoP3.history.convergenceCount(end)==3, ...
    'Converged with a persistence counter of %d, expected 3.',infoP3.history.convergenceCount(end));
assert(infoP3.iterations.outer==3,'Convergence was declared before the persistence was met.');
rcConv.persistence=8;rcConv.max_outer_iterations=5;
[~,~,infoP8]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply',rcConv);
assert(strcmp(infoP8.status,'CAP_HIT'), ...
    'A run capped below its persistence was classified %s.',infoP8.status);
assert(max(infoP8.history.convergenceCount)<8,'Persistence was reached despite the cap.');
nPass=nPass+1;

%% 7. the convergence tolerance is clamped to objective_tol/certificate_radius
rcLoose=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
    'max_outer_iterations',2,'max_trial_steps',2,'stationarity_tol',1e6, ...
    'objective_tol',1e-5,'move_max',.005);
[~,~,infoLoose]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply',rcLoose);
r=infoLoose.regularization;
bound=r.objectiveTol/r.certRadius;
assert(r.convergenceStationarityTol<=bound*(1+1e-12), ...
    'A loose stationarity_tol leaked into the convergence test.');
assert(abs(r.convergenceStationarityTol-bound)<=1e-12*bound, ...
    'The clamped convergence tolerance is not objective_tol/certificate_radius.');
assert(r.stationarityTol<=bound*(1+1e-12), ...
    'A loose stationarity_tol leaked into the move-ceiling contraction gate.');
assert(r.requestedStationarityTol==1e6,'The caller''s request was not recorded.');
%   ... and behaviourally: a loose request must not let the ceiling contract at a
%   point the convergence test would reject.  A contraction gate looser than the
%   convergence gate collapses the ceiling onto move_min and guarantees a stall.
rcSplit=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
    'max_outer_iterations',8,'max_trial_steps',2,'stationarity_tol',1e6, ...
    'objective_tol',1e-5,'progress_tolerance',1e9,'progress_spike_tolerance',1e9, ...
    'progress_window',1,'progress_dwell',1,'persistence',1000);
[~,~,infoSplit]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply',rcSplit);
assert(min(infoSplit.history.stationarityMeasure)>infoSplit.regularization.convergenceStationarityTol, ...
    'The toy case was already stationary; the split test proves nothing.');
assert(infoSplit.iterations.move_ceiling_contractions==0, ...
    'The move ceiling contracted at a point the convergence test rejects (CV-4).');
nPass=nPass+1;

%% 8. the certificate is radius-scale invariant and never follows the trust --
rcA=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
    'max_outer_iterations',1,'max_trial_steps',2,'certificate_radius',5e-3);
rcB=rcA;rcB.certificate_radius=5e-4;
[~,~,infoA]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply',rcA);
[~,~,infoB]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply',rcB);
sA=infoA.history.certificateSlope(1);sB=infoB.history.certificateSlope(1);
assert(sA>0&&sB>0,'The certificate reported no ascent at the uniform initial design.');
assert(abs(sA-sB)<=1e-6*sA, ...
    'The certificate is not radius-scale invariant (%.6e vs %.6e); a shrinking radius could manufacture stationarity.',sA,sB);
rcC=rcForce;rcC.max_outer_iterations=6;rcC.progress_window=1;rcC.progress_dwell=1;
[~,~,infoC]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'fixedPinned',rcC);
hc=infoC.history;rc2=infoC.regularization;
assert(min(hc.trustUsed)<max(hc.trustUsed)||infoC.iterations.move_ceiling_contractions>=1, ...
    'The trust/ceiling never moved, so trust independence was not exercised.');
assert(max(abs(hc.certificateRelativeGain-hc.certificateSlope*rc2.certRadius)) ...
    <=1e-9*max(1,max(hc.certificateRelativeGain)), ...
    'The certificate was not evaluated at the fixed reference radius.');
nPass=nPass+1;

%% 9. the certificate uses the EXACT multiplicity, not cfg.tolMult -----------
rcMult=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
    'max_outer_iterations',1,'max_trial_steps',2,'tol_mult',3.0, ...
    'certificate_mult_tol',1e-12);
[~,~,infoM]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply',rcMult);
assert(infoM.history.N(1)>1,'The step model did not cluster under tol_mult=3.');
assert(infoM.history.certificateN(1)==1, ...
    'The certificate clustered strictly separated modes (certificateN=%d).',infoM.history.certificateN(1));
%   ... and the cluster must be SELF-CONSISTENT with its own prediction
%   (CV-1b): a certificate that predicts lambda_n rising past the first
%   EXCLUDED eigenvalue is not a model of the ordered lambda_n over that step.
%   The fixed point is  predicted gain <= gap to the first excluded mode,
%   unless the cluster has already reached Nmax or the LP failed.  Assert the
%   invariant over whole runs, on every route and at radii three orders of
%   magnitude apart, rather than a single contrived growth event.
for r=[1e-4 5e-3 1]
    for bc={'simply','fixedPinned'}
        rcInv=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
            'max_outer_iterations',6,'max_trial_steps',2,'certificate_radius',r, ...
            'move_max',max(r,.005),'certificate_mult_tol',1e-12,'objective_tol',1e9);
        [~,~,infoInv]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,bc{1},rcInv);
        hi=infoInv.history;ok=hi.certificateLpFlag~=1|hi.certificateN>=infoInv.cfg.Nmax| ...
            hi.certificateRelativeGain<=hi.certificateNextGap.*(1+1e-9);
        assert(all(ok),['Certificate cluster is not self-consistent for %s at ' ...
            'radius %g: gain %.4e exceeds the gap %.4e to the excluded mode at N=%d.'], ...
            bc{1},r,max(hi.certificateRelativeGain(~ok)), ...
            min(hi.certificateNextGap(~ok)),max(hi.certificateN(~ok)));
        assert(all(hi.certificateN>=1&hi.certificateN<=infoInv.cfg.Nmax), ...
            'Certificate multiplicity left [1,Nmax].');
        assert(all(hi.certificateGrown>=0),'Certificate growth counter is negative.');
    end
end
nPass=nPass+1;

%   ... and a local model that returns a NEGATIVE predicted improvement must
%   never be read as stationarity (CV-5).  drho = 0 is always feasible, so a
%   solved subproblem cannot do this; when it happens the measure must be +Inf,
%   not 0.  Asserted structurally over every recorded iteration of every route:
%   a zero route slope may only appear where the model actually returned zero.
for i=1:size(routes,1)
    rcNeg=struct('verbose',false,'formulation',routes{i,1},'optimizer',routes{i,2}, ...
        'max_outer_iterations',4,'max_inner_iterations',60,'min_inner',3, ...
        'max_trial_steps',2);
    [~,~,infoNeg]=topopt_olhoff_regularized(16,2,.5,3,1.3,.005,'simply',rcNeg);
    hn=infoNeg.history;
    bad=hn.predictedImprovement<0 & isfinite(hn.predictedSlope);
    assert(~any(bad), ...
        ['Route %s/%s reported a finite stationarity slope for a negative ' ...
         'predicted improvement (%.3e) -- a failed local model was read as ' ...
         'stationarity.'],routes{i,1},routes{i,2},min(hn.predictedImprovement(bad)));
    assert(isfield(infoNeg.iterations,'negative_predictions'), ...
        'Negative predicted improvements are not counted.');
end
nPass=nPass+1;

%% 10. cantilever concentrated-mass path -------------------------------------
rcCant=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
    'max_outer_iterations',1,'max_inner_iterations',10,'max_trial_steps',2);
[rho,w,infoCant]=topopt_olhoff_regularized(8,4,.5,3,1.3,.005,'cantilever',rcCant);
assert(infoCant.model.tipMassValue>0&&all(isfinite(rho))&&all(isfinite(w)), ...
    'Cantilever point-mass smoke failed.');
nPass=nPass+1;

%% 11. every runner honours an externally selected route ---------------------
%   Regression for the fixed--pinned runner, which used to overwrite the
%   caller's optimizer with a literal after advertising the override.
here=fileparts(mfilename('fullpath'));
runners={'run_regularized_simply_supported.m','run_regularized_fixed_pinned.m', ...
         'run_regularized_cantilever.m'};
for i=1:numel(runners)
    src=fileread(fullfile(here,runners{i}));
    lines=regexp(src,'\r?\n','split');
    for j=1:numel(lines)
        L=strtrim(regexprep(lines{j},'%.*$',''));
        if isempty(L),continue,end
        if ~isempty(regexp(L,'^(optimizer|formulation)\s*=','once'))
            error('OlhoffRegularizedTests:RunnerOverwritesRoute', ...
                '%s line %d unconditionally overwrites the caller''s route: %s', ...
                runners{i},j,L);
        end
    end
    assert(contains(src,'~exist(''optimizer'',''var'')')&& ...
           contains(src,'~exist(''formulation'',''var'')'), ...
        '%s no longer advertises an externally selectable route.',runners{i});
end
nPass=nPass+1;

fprintf('OLHOFF_REGULARIZED_TESTS_PASS groups=%d\n',nPass);
end
