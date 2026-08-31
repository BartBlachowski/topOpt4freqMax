function extract_olhoff_evidence()
%EXTRACT_OLHOFF_EVIDENCE Read-only postprocessing of frozen campaign MAT files.
% This function never calls an optimizer or LP solver and never writes below
% examples/Performance/final_campaign.  It extracts retained histories,
% evaluates three already-saved best-prior density fields with the frozen
% common evaluator, and writes only to this audit directory.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
frozen = fullfile(repo, 'examples', 'Performance', 'final_campaign', 'raw', 'olhoff');
histDir = fullfile(here, 'olhoff_histories');
topoDir = fullfile(here, 'olhoff_topologies');
figDir = fullfile(here, 'figures');
if exist(histDir, 'dir') ~= 7, mkdir(histDir); end
if exist(topoDir, 'dir') ~= 7, mkdir(topoDir); end
if exist(figDir, 'dir') ~= 7, mkdir(figDir); end
addpath(fullfile(repo, 'analysis', 'three_method_parametric_study'));
maxNumCompThreads(1);

meshes = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
summaryRows = cell(size(meshes,1), 36);
bestRows = cell(3, 32);
failureMeshes = [480 60; 560 70; 640 80];

allHistory = table();
failureData = cell(3,1);
neighborhoodData = cell(5,1);
neighborhoodMeshes = [400 50; 480 60; 560 70; 640 80; 720 90];

for i = 1:size(meshes,1)
    nelx = meshes(i,1); nely = meshes(i,2);
    mesh = sprintf('%dx%d', nelx, nely);
    source = fullfile(frozen, sprintf('s1_%s.mat', mesh));
    data = load(source, 'res');
    r = data.res; h = r.hist; n = r.nOuter;
    iter = (1:n)';
    rhoSteps = double(r.rho_snapshots);
    turnover = NaN(n,1);
    if size(rhoSteps,2) >= n + 1
        for k = 1:n
            turnover(k) = mean((rhoSteps(:,k) >= 0.5) ~= (rhoSteps(:,k+1) >= 0.5));
        end
    end

    H = table(repmat(string(mesh),n,1), iter, h.omega(1,:)', h.omega(2,:)', ...
        h.omega(3,:)', h.N(:), h.gap12(:), h.moveLimit(:), h.policyStage(:), ...
        logical(h.trigger(:)), h.dxOuter(:), h.dRms(:), h.vol(:), ...
        h.volumeResidual(:), h.moveBoundFraction(:), h.stronglyMovingFraction(:), ...
        h.lpFlag(:), logical(h.finiteOk(:)), h.beta(:), h.tEig(:), h.tGrad(:), ...
        h.tInner(:), turnover, ...
        'VariableNames', {'mesh','iteration','omega1','omega2','omega3','N','gap12', ...
        'move','policy_stage','trigger','max_dx','rms_dx','volume','volume_residual', ...
        'move_bound_fraction','strongly_moving_fraction','lp_flag','finite_ok','beta', ...
        't_eig_s','t_grad_s','t_inner_s','binary_turnover'});
    writetable(H, fullfile(histDir, sprintf('%s.csv', mesh)));
    allHistory = [allHistory; H]; %#ok<AGROW>

    [bestOmega1, bestIter] = max(h.omega(1,:));
    bestRho = double(r.rho_snapshots(:,bestIter)); % h(:,k) evaluates x_{k-1}
    finalRho = double(r.rho(:));
    bestGray = mean(4*bestRho.*(1-bestRho));
    finalGray = mean(4*finalRho.*(1-finalRho));
    bestConn = simple_connectivity(bestRho, nely, nelx);
    finalConn = simple_connectivity(finalRho, nely, nelx);
    q = quantile(finalRho, [0 .01 .1 .5 .9 .99 1]);
    late = max(1,n-49):n;
    nSwitches = sum(diff(h.N(late)) ~= 0);
    gapCrossings = sum(diff(h.gap12(late) <= 0.01) ~= 0);
    failureAttempt = r.failure_iteration;
    if isnan(failureAttempt), failureAttempt = NaN; end
    triggerIter = NaN;
    if ~isempty(r.trigger_iterations), triggerIter = r.trigger_iterations(1); end
    logText = strjoin(r.log, ' | ');
    summaryRows(i,:) = {mesh, nelx, nely, char(r.status), n, failureAttempt, triggerIter, ...
        r.final_policy_stage, h.moveLimit(end), h.N(end), h.omega(1,end), h.omega(2,end), ...
        h.omega(3,end), h.gap12(end), h.lpFlag(end), h.finiteOk(end), h.vol(end), ...
        finalGray, finalConn.left_right_connected, finalConn.n_components, ...
        finalConn.largest_component_fraction, bestIter, bestOmega1, h.omega(2,bestIter), ...
        h.omega(3,bestIter), h.N(bestIter), h.gap12(bestIter), bestGray, ...
        bestConn.left_right_connected, bestConn.n_components, std(h.omega(1,late)), ...
        std(h.omega(3,late)), nSwitches, gapCrossings, max(turnover(late)), logText};

    fig = figure('Visible','off','Color','w','Position',[100 100 1100 260]);
    imagesc(reshape(1-finalRho,nely,nelx)); axis image off; colormap(gray(256));
    title(sprintf('Olhoff %s: %s, completed k=%d', mesh, r.status, n), ...
        'Interpreter','none');
    exportgraphics(fig, fullfile(topoDir, sprintf('olhoff_%s_final.png', mesh)), 'Resolution', 180);
    close(fig);

    idxN = find(neighborhoodMeshes(:,1)==nelx & neighborhoodMeshes(:,2)==nely,1);
    if ~isempty(idxN), neighborhoodData{idxN} = H; end

    idxF = find(failureMeshes(:,1)==nelx & failureMeshes(:,2)==nely,1);
    if ~isempty(idxF)
        fprintf('Common-evaluating saved best-prior Olhoff state %s at k=%d\n', mesh, bestIter);
        ev = study_evaluate_design(bestRho, nelx, nely, 0.5);
        wr1=ev.omega_raw_E1; wr2=ev.omega_raw_E2; wr3=ev.omega_raw_E3;
        wb1=ev.omega_binary_E1; wb2=ev.omega_binary_E2; wb3=ev.omega_binary_E3;
        bestRows(idxF,:) = {mesh,bestIter,bestOmega1,h.omega(2,bestIter),h.omega(3,bestIter), ...
            wr1(1),wr1(2),wr1(3),wr2(1),wr2(2),wr2(3),wr3(1),wr3(2),wr3(3), ...
            wb1(1),wb1(2),wb1(3),wb2(1),wb2(2),wb2(3),wb3(1),wb3(2),wb3(3), ...
            h.N(bestIter),h.gap12(bestIter),mean(bestRho),bestGray, ...
            ev.connectivity_raw_05.left_right_connected,ev.connectivity_binary.left_right_connected, ...
            ev.connectivity_raw_05.largest_component_fraction, ...
            ev.connectivity_binary.largest_component_fraction, source};
        failureData{idxF} = struct('mesh',mesh,'rho',bestRho,'nely',nely,'nelx',nelx, ...
            'bestIter',bestIter,'omega1',bestOmega1);
    end
end

writetable(allHistory, fullfile(here, 'olhoff_histories.csv'));

summaryNames = {'mesh','nelx','nely','runner_status','completed_iterations','failure_attempt', ...
    'stabilization_trigger_iteration','final_policy_stage','final_move','last_valid_N', ...
    'last_valid_omega1','last_valid_omega2','last_valid_omega3','last_valid_gap12', ...
    'last_recorded_lp_flag','last_recorded_finite_ok','final_volume','final_grayness', ...
    'final_connected','final_component_count','final_largest_component_fraction', ...
    'best_native_iteration','best_native_omega1','best_native_omega2','best_native_omega3', ...
    'best_native_N','best_native_gap12','best_native_grayness','best_native_connected', ...
    'best_native_component_count','late50_omega1_std','late50_omega3_std','late50_N_switches', ...
    'late50_gap_threshold_crossings','late50_max_binary_turnover','runner_log'};
summary = cell2table(summaryRows, 'VariableNames', summaryNames);
writetable(summary, fullfile(here, 'olhoff_raw_summary.csv'));

bestNames = {'mesh','iteration','native_omega1','native_omega2','native_omega3', ...
    'raw_E1_omega1','raw_E1_omega2','raw_E1_omega3','raw_E2_omega1','raw_E2_omega2', ...
    'raw_E2_omega3','raw_E3_omega1','raw_E3_omega2','raw_E3_omega3', ...
    'binary_E1_omega1','binary_E1_omega2','binary_E1_omega3','binary_E2_omega1', ...
    'binary_E2_omega2','binary_E2_omega3','binary_E3_omega1','binary_E3_omega2', ...
    'binary_E3_omega3','N','gap12','volume','grayness','connected_raw','connected_binary', ...
    'largest_component_fraction_raw','largest_component_fraction_binary','source_mat'};
best = cell2table(bestRows, 'VariableNames', bestNames);
writetable(best, fullfile(here, 'olhoff_best_prior_quality.csv'));

make_failure_history_figure(neighborhoodData, figDir);
make_best_topology_figure(failureData, figDir, topoDir);
fprintf('Olhoff evidence extraction complete.\n');
end

function c = simple_connectivity(rho,nely,nelx)
B=reshape(rho,nely,nelx)>=0.5; visited=false(size(B)); labels=zeros(size(B)); sizes=[]; component=0;
for rr=1:nely
    for cc=1:nelx
        if ~B(rr,cc)||visited(rr,cc),continue;end
        component=component+1; qr=zeros(nnz(B),1); qc=qr; head=1; tail=1;
        qr(1)=rr;qc(1)=cc;visited(rr,cc)=true;count=0;
        while head<=tail
            r0=qr(head);c0=qc(head);head=head+1;count=count+1;labels(r0,c0)=component;
            nbr=[r0-1 c0;r0+1 c0;r0 c0-1;r0 c0+1];
            for k=1:4
                r1=nbr(k,1);c1=nbr(k,2);
                if r1>=1&&r1<=nely&&c1>=1&&c1<=nelx&&B(r1,c1)&&~visited(r1,c1)
                    tail=tail+1;qr(tail)=r1;qc(tail)=c1;visited(r1,c1)=true;
                end
            end
        end
        sizes(component)=count; %#ok<AGROW>
    end
end
left=unique(labels(:,1));left(left==0)=[];right=unique(labels(:,end));right(right==0)=[];
c=struct('n_components',component,'left_right_connected',~isempty(intersect(left,right)), ...
    'largest_component_fraction',max([0 sizes])/max(1,nnz(B)));
end

function make_failure_history_figure(data, figDir)
fig=figure('Visible','off','Color','w','Position',[100 100 1500 950]);
colors=lines(numel(data)); labels=cell(numel(data),1);
for i=1:numel(data)
    T=data{i}; labels{i}=char(T.mesh(1));
    subplot(2,2,1); hold on; plot(T.iteration,T.omega1,'Color',colors(i,:));
    subplot(2,2,2); hold on; plot(T.iteration,T.omega3,'Color',colors(i,:));
    subplot(2,2,3); hold on; semilogy(T.iteration,max(T.gap12,1e-8),'Color',colors(i,:));
    subplot(2,2,4); hold on; plot(T.iteration,T.N,'Color',colors(i,:));
end
subplot(2,2,1); ylabel('\omega_1 [rad/s]'); xlabel('completed iteration'); grid on; legend(labels,'Location','best');
subplot(2,2,2); ylabel('\omega_3 [rad/s]'); xlabel('completed iteration'); grid on;
subplot(2,2,3); ylabel('gap_{12}'); xlabel('completed iteration'); yline(.01,'k--'); grid on;
subplot(2,2,4); ylabel('N'); xlabel('completed iteration'); grid on; ylim([0.5 3.5]);
sgtitle('Olhoff failure neighborhood: retained histories (failed LP attempt itself is not logged)');
exportgraphics(fig,fullfile(figDir,'09_olhoff_failure_neighborhood_histories.png'),'Resolution',180);
close(fig);
end

function make_best_topology_figure(data, figDir, topoDir)
fig=figure('Visible','off','Color','w','Position',[100 100 1400 520]);
for i=1:numel(data)
    d=data{i}; subplot(numel(data),1,i);
    imagesc(reshape(1-d.rho,d.nely,d.nelx));axis image off;colormap(gray(256));
    title(sprintf('%s best valid native state: k=%d, \\omega_1=%.3f',d.mesh,d.bestIter,d.omega1));
    f2=figure('Visible','off','Color','w','Position',[100 100 1100 260]);
    imagesc(reshape(1-d.rho,d.nely,d.nelx));axis image off;colormap(gray(256));
    title(sprintf('Olhoff %s best prior k=%d, \\omega_1=%.3f',d.mesh,d.bestIter,d.omega1));
    exportgraphics(f2,fullfile(topoDir,sprintf('olhoff_%s_best_prior.png',d.mesh)),'Resolution',180);close(f2);
end
sgtitle('Best valid pre-failure Olhoff density fields');
exportgraphics(fig,fullfile(figDir,'11_olhoff_best_prior_topologies.png'),'Resolution',180);
close(fig);
end
