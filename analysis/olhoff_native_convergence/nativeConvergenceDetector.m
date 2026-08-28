function [fire,raw] = nativeConvergenceDetector(hist,t,k,cfg,d)
%NATIVECONVERGENCEDETECTOR Future-blind online Du--Olhoff stop detector.
% Solver/finite-state health and intended N=2 modal structure are mandatory.

required={'objective_block','window','persistence','objective_block_drift_tol', ...
    'objective_phase_recurrence_tol','rho_phase_rms_tol', ...
    'topology_phase_turnover_tol','modal_window','gap_tol','volume_tol_rel', ...
    'required_N'};
for i=1:numel(required)
    if ~isfield(d,required{i})
        error('nativeConvergenceDetector:MissingField','Missing detector field %s',required{i});
    end
end
q=max([2*d.objective_block d.window d.modal_window]);
% The phase-recurrence term reads omega(ix-2) over the window, so the earliest
% iteration at which this condition is DEFINED is window+2, not q.  Before this
% guard the function raised a subscript error for configurations with
% q == window (the frozen H_balanced_v1 is one).  Every k at which the old code
% returned a value returns the same value here: the guard only covers k that
% previously errored.
qmin=max(q,d.window+2);
raw=false(1,max(k,1));
if k<qmin+d.persistence-1, fire=false; return; end
for kk=max(qmin,k-d.persistence+1):k
    B=d.objective_block;
    newMean=mean(hist.omega(1,kk-B+1:kk));
    oldMean=mean(hist.omega(1,kk-2*B+1:kk-B));
    blockDrift=abs(newMean-oldMean)/max(abs(newMean),eps);
    ix=kk-d.window+1:kk;
    phaseRecurrence=abs(hist.omega(1,ix)-hist.omega(1,ix-2))./ ...
        max(abs(hist.omega(1,ix)),eps);
    im=kk-d.modal_window+1:kk;
    modalOK=all(hist.N(im)==d.required_N)&&all(t.gaps_rel(1,im)<=d.gap_tol)&& ...
        ~any(t.mode_order_changed(im))&&~any(t.N_changed(im));
    healthOK=all(hist.innerConv(im))&&all(t.lp_flag(im)==1)&&all(t.eig_ok(im))&& ...
        all(t.finite_ok(im))&&~any(t.eig_warning(im));
    volumeOK=abs(hist.vol(kk)-cfg.volfrac)/cfg.volfrac<=d.volume_tol_rel;
    raw(kk)=blockDrift<=d.objective_block_drift_tol&& ...
        max(phaseRecurrence)<=d.objective_phase_recurrence_tol&& ...
        max(t.rho_phase_rms(ix))<=d.rho_phase_rms_tol&& ...
        max(t.topology_phase_turnover(ix))<=d.topology_phase_turnover_tol&& ...
        modalOK&&healthOK&&volumeOK;
end
fire=all(raw(k-d.persistence+1:k));
end
