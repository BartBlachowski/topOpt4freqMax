function run_checkpoint_audit()
%RUN_CHECKPOINT_AUDIT Offline fixed-budget audit on frozen density histories.
% This script never calls the optimizer and never changes its numerical code.

maxNumCompThreads(1);
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo,'analysis','three_method_parametric_study'));

pre = jsondecode(fileread(fullfile(here,'study_preregistration.json')));
checkpoints = double(pre.predeclared_checkpoints(:));
meshes = [[pre.meshes.nelx]' [pre.meshes.nely]'];
rows = repmat(empty_row(),numel(checkpoints)*size(meshes,1),1);
modalRows = repmat(empty_modal_row(),4*size(meshes,1),1);
raw = struct();
ri = 0; mi = 0;

for im = 1:size(meshes,1)
    nelx = meshes(im,1); nely = meshes(im,2);
    meshId = sprintf('%dx%d',nelx,nely);
    src = fullfile(repo,'analysis','olhoff_native_convergence','results', ...
        sprintf('development_%s.mat',meshId));
    S = load(src,'res','identity'); res = S.res;
    assert(res.nOuter == 1600 && res.telemetry.snapshot_stride == 1);
    assert(res.cfg.move == 0.005 && res.cfg.rminEl == 1.3 && ...
        res.cfg.tolMult == 0.05 && res.cfg.rhomin == 0.001 && ...
        strcmp(res.cfg.innerSolver,'lp') && strcmp(res.cfg.filterMode,'diag') && ...
        ~res.cfg.offDiag,'Frozen profile mismatch.');
    assert(all(res.telemetry.lp_flag == 1) && all(res.telemetry.finite_ok), ...
        'Unhealthy saved trajectory.');

    X = double(res.telemetry.rho_snapshots);
    xRef = X(:,1601);
    xbRef = volume_binary(xRef,0.5);
    native = aligned_native(res);
    gap = (native.omega2-native.omega1)./native.omega1;
    nSeries = native.N;
    modalClass200 = classify_modal_200(nSeries,gap);

    conditions = {nSeries==2, gap<=0.05, gap<=0.02, gap<=0.01};
    names = {'N_EQ_2','GAP12_LE_5PCT','GAP12_LE_2PCT','GAP12_LE_1PCT'};
    for ic = 1:numel(conditions)
        mi = mi+1; c = conditions{ic};
        [first,persistentEntry,nClosures,nOpenings] = entries(c);
        modalRows(mi).mesh = string(meshId);
        modalRows(mi).condition = string(names{ic});
        modalRows(mi).first_entry = first;
        modalRows(mi).persistent_entry = persistentEntry;
        modalRows(mi).disappears_after_first = first>=0 && any(~c(first+1:end));
        modalRows(mi).n_later_closures = nClosures;
        modalRows(mi).n_later_openings = nOpenings;
        modalRows(mi).k200_condition = c(201);
        modalRows(mi).k200_classification = string(modalClass200);
    end

    loopTime = res.hist.tEig + res.hist.tGrad + res.hist.tInner;
    raw.(sprintf('m_%d_%d',nelx,nely)) = struct( ...
        'source',src,'native',native,'gap12',gap,'loop_time',loopTime, ...
        'density_terminal',xRef,'binary_terminal',xbRef);

    initialOmega = native.omega1(1);
    terminalOmega = native.omega1(end);
    for ik = 1:numel(checkpoints)
        k = checkpoints(ik); ri = ri+1;
        x = X(:,k+1); xb = volume_binary(x,0.5);
        ev = study_evaluate_design(x,nelx,nely,0.5);
        rows(ri) = make_row(meshId,nelx,nely,k,x,xb,xRef,xbRef,ev, ...
            native,gap,nSeries,loopTime,res,initialOmega,terminalOmega);
        rows(ri).modal_k200_classification = string(modalClass200);
        fprintf('checkpoint %s k=%d E1raw=%.9f loss pending\n', ...
            meshId,k,ev.omega_raw_E1(1));
    end
end

T = struct2table(rows);
for im = 1:size(meshes,1)
    meshId = sprintf('%dx%d',meshes(im,1),meshes(im,2));
    ix = T.mesh == meshId; ref = T(ix & T.iteration==1600,:);
    reps = {'raw','binary'}; evals = {'E1','E2','E3'};
    for ir = 1:numel(reps)
        for ie = 1:numel(evals)
            col = sprintf('common_%s_%s_omega1',reps{ir},evals{ie});
            qcol = sprintf('%s_ratio_to_k1600',col);
            lcol = sprintf('%s_loss_to_k1600',col);
            T.(qcol)(ix) = T.(col)(ix)./ref.(col);
            T.(lcol)(ix) = (ref.(col)-T.(col)(ix))./ref.(col);
        end
    end
end
writetable(T,fullfile(here,'checkpoint_metrics.csv'));
writetable(struct2table(modalRows),fullfile(here,'modal_establishment.csv'));
save(fullfile(here,'raw','checkpoint_evidence.mat'),'raw','T','modalRows','pre','-v7.3');
fprintf('Wrote %d checkpoint rows and %d modal rows.\n',height(T),numel(modalRows));
end

function r = make_row(meshId,nelx,nely,k,x,xb,xRef,xbRef,ev,native,gap,N,loopTime,res,w0,wT)
r = empty_row(); r.mesh=string(meshId); r.nelx=nelx; r.nely=nely; r.iteration=k;
r.native_omega1=native.omega1(k+1); r.native_omega2=native.omega2(k+1);
r.native_omega3=native.omega3(k+1); r.native_gap12=gap(k+1); r.native_N=N(k+1);
r.modal_event_state=string(modal_state(N(k+1),gap(k+1)));
r.native_omega1_ratio_to_k1600=r.native_omega1/native.omega1(end);
r.native_omega1_loss_to_k1600=(native.omega1(end)-r.native_omega1)/native.omega1(end);
r.native_improvement_fraction=(r.native_omega1-w0)/(wT-w0);
r.volume=ev.volume; r.volume_residual=ev.volume_residual;
r.grayness=ev.grayness; r.gray_fraction_01_09=ev.gray_fraction_01_09;
r.raw_05_n_components=ev.connectivity_raw_05.n_components;
r.raw_05_left_right_connected=ev.connectivity_raw_05.left_right_connected;
r.raw_05_largest_component_fraction=ev.connectivity_raw_05.largest_component_fraction;
r.binary_n_components=ev.connectivity_binary.n_components;
r.binary_left_right_connected=ev.connectivity_binary.left_right_connected;
r.binary_largest_component_fraction=ev.connectivity_binary.largest_component_fraction;
r.density_rms_to_k1600=sqrt(mean((x-xRef).^2));
r.binary_turnover_to_k1600=mean(xb~=xbRef);
r.common_raw_E1_omega1=ev.omega_raw_E1(1); r.common_raw_E2_omega1=ev.omega_raw_E2(1); r.common_raw_E3_omega1=ev.omega_raw_E3(1);
r.common_binary_E1_omega1=ev.omega_binary_E1(1); r.common_binary_E2_omega1=ev.omega_binary_E2(1); r.common_binary_E3_omega1=ev.omega_binary_E3(1);
r.common_raw_E1_omega2=ev.omega_raw_E1(2); r.common_raw_E1_omega3=ev.omega_raw_E1(3);
r.cumulative_loop_time_s=sum(loopTime(1:k)); r.mean_loop_time_per_iteration_s=r.cumulative_loop_time_s/k;
lo=max(1,k-49); r.window50_loop_time_per_iteration_s=mean(loopTime(lo:k));
r.cumulative_eigensolve_time_s=sum(res.hist.tEig(1:k));
r.cumulative_eigensolve_share=r.cumulative_eigensolve_time_s/r.cumulative_loop_time_s;
r.lp_failures_through_checkpoint=nnz(res.telemetry.lp_flag(1:k)~=1);
r.nonfinite_iterations_through_checkpoint=nnz(~res.telemetry.finite_ok(1:k));
end

function n = aligned_native(res)
% State x_k is evaluated at the start of update k+1; terminal x_1600 has
% the optimizer's post-loop modal solve. Arrays therefore cover k=0:1600.
n.omega1=[res.hist.omega(1,:) res.omega(1)]';
n.omega2=[res.hist.omega(2,:) res.omega(2)]';
n.omega3=[res.hist.omega(3,:) res.omega(3)]';
n.N=[res.hist.N detect_n(res.omega,res.cfg)]';
end

function N = detect_n(w,cfg)
N=1;
while cfg.n+N <= numel(w)-1 && abs(w(cfg.n+N)-w(cfg.n))/w(cfg.n) < cfg.tolMult
    N=N+1;
end
end

function cls = classify_modal_200(N,gap)
c = N==2 & gap<=0.01;
if ~c(201), cls='BIMODAL_NOT_YET_ESTABLISHED';
elseif all(c(201:end)), cls='BIMODAL_ESTABLISHED';
else, cls='BIMODAL_TRANSIENT';
end
end

function s = modal_state(N,gap)
if N==2 && gap<=0.01, s='N2_GAP_LE_1PCT';
elseif N==2 && gap<=0.02, s='N2_GAP_LE_2PCT';
elseif N==2 && gap<=0.05, s='N2_GAP_LE_5PCT';
elseif N==2, s='N2_GAP_GT_5PCT';
else, s=sprintf('N%d',N);
end
end

function [first,persistentEntry,nClose,nOpen] = entries(c)
ix=find(c,1,'first'); if isempty(ix), first=NaN; else, first=ix-1; end
bad=find(~c); lastBad=max([-1; bad(:)-1]);
if lastBad>=numel(c)-1, persistentEntry=NaN; else, persistentEntry=lastBad+1; end
d=diff(c(:)); nClose=nnz(d==-1); nOpen=nnz(d==1);
end

function xb = volume_binary(x,volfrac)
nSolid=round(volfrac*numel(x)); [~,order]=sortrows([-x(:),(1:numel(x))'],[1 2]);
xb=false(size(x)); xb(order(1:nSolid))=true;
end

function r = empty_row()
r=struct('mesh',string(missing),'nelx',NaN,'nely',NaN,'iteration',NaN, ...
    'native_omega1',NaN,'native_omega2',NaN,'native_omega3',NaN,'native_gap12',NaN,'native_N',NaN, ...
    'modal_event_state',string(missing),'modal_k200_classification',string(missing), ...
    'native_omega1_ratio_to_k1600',NaN,'native_omega1_loss_to_k1600',NaN,'native_improvement_fraction',NaN, ...
    'volume',NaN,'volume_residual',NaN,'grayness',NaN,'gray_fraction_01_09',NaN, ...
    'raw_05_n_components',NaN,'raw_05_left_right_connected',false,'raw_05_largest_component_fraction',NaN, ...
    'binary_n_components',NaN,'binary_left_right_connected',false,'binary_largest_component_fraction',NaN, ...
    'density_rms_to_k1600',NaN,'binary_turnover_to_k1600',NaN, ...
    'common_raw_E1_omega1',NaN,'common_raw_E2_omega1',NaN,'common_raw_E3_omega1',NaN, ...
    'common_binary_E1_omega1',NaN,'common_binary_E2_omega1',NaN,'common_binary_E3_omega1',NaN, ...
    'common_raw_E1_omega2',NaN,'common_raw_E1_omega3',NaN, ...
    'common_raw_E1_omega1_ratio_to_k1600',NaN,'common_raw_E1_omega1_loss_to_k1600',NaN, ...
    'common_raw_E2_omega1_ratio_to_k1600',NaN,'common_raw_E2_omega1_loss_to_k1600',NaN, ...
    'common_raw_E3_omega1_ratio_to_k1600',NaN,'common_raw_E3_omega1_loss_to_k1600',NaN, ...
    'common_binary_E1_omega1_ratio_to_k1600',NaN,'common_binary_E1_omega1_loss_to_k1600',NaN, ...
    'common_binary_E2_omega1_ratio_to_k1600',NaN,'common_binary_E2_omega1_loss_to_k1600',NaN, ...
    'common_binary_E3_omega1_ratio_to_k1600',NaN,'common_binary_E3_omega1_loss_to_k1600',NaN, ...
    'cumulative_loop_time_s',NaN,'mean_loop_time_per_iteration_s',NaN,'window50_loop_time_per_iteration_s',NaN, ...
    'cumulative_eigensolve_time_s',NaN,'cumulative_eigensolve_share',NaN, ...
    'lp_failures_through_checkpoint',NaN,'nonfinite_iterations_through_checkpoint',NaN);
end

function r = empty_modal_row()
r=struct('mesh',string(missing),'condition',string(missing),'first_entry',NaN, ...
    'persistent_entry',NaN,'disappears_after_first',false,'n_later_closures',NaN, ...
    'n_later_openings',NaN,'k200_condition',false,'k200_classification',string(missing));
end
