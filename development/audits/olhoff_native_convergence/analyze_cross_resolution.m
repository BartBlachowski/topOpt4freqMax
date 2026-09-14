function analyze_cross_resolution()
%ANALYZE_CROSS_RESOLUTION Locked, no-retuning replay on four meshes.
here=fileparts(mfilename('fullpath')); resultDir=fullfile(here,'results');
cfg=jsondecode(fileread(fullfile(resultDir,'native_convergence_config.json')));
detectors={cfg.selected_detector,cfg.predeclared_sensitivity_detector};
meshes=[160 20;240 30;320 40;400 50];
roles={'holdout';'development';'holdout';'holdout'};
rows=cell(size(meshes,1)*numel(detectors),1); rr=0;

allRuns=cell(size(meshes,1),1); allFires=NaN(size(meshes,1),numel(detectors));
for m=1:size(meshes,1)
    fn=fullfile(resultDir,sprintf('development_%dx%d.mat',meshes(m,1),meshes(m,2)));
    s=load(fn,'res'); r=s.res; allRuns{m}=r;
    [rhoPhase,topoPhase]=localPhaseMetrics(r);
    for c=1:numel(detectors)
        d=detectors{c}; k=localFirstFire(r,d,rhoPhase,topoPhase);
        allFires(m,c)=k; rr=rr+1;
        rows{rr}=localRow(r,meshes(m,:),roles{m},d,k,rhoPhase,topoPhase);
    end
end
cross=struct2table(vertcat(rows{:}));
writetable(cross,fullfile(resultDir,'native_convergence_cross_resolution.csv'));

% Plot 1: spectra with locked primary firing points.
fig=figure('Visible','off','Color','w','Position',[100 100 1100 850]);
tiledlayout(2,2,'TileSpacing','compact');
for m=1:4
    r=allRuns{m}; nexttile; plot(r.hist.omega(1:3,:)'); grid on; hold on
    if isfinite(allFires(m,1)), xline(allFires(m,1),'k--','H balanced'); end
    title(sprintf('%dx%d',meshes(m,1),meshes(m,2))); ylabel('\omega'); xlabel('iteration');
end
legend('\omega_1','\omega_2','\omega_3','Location','best');
exportgraphics(fig,fullfile(resultDir,'cross_resolution_spectra_and_firing.png'),'Resolution',180); close(fig)

% Plot 2: eigengap and multiplicity.
fig=figure('Visible','off','Color','w','Position',[100 100 1100 850]);
tiledlayout(2,2,'TileSpacing','compact');
for m=1:4
    r=allRuns{m}; nexttile; yyaxis left; semilogy(max(r.telemetry.gaps_rel(1,:),1e-8));
    ylabel('relative gap 1--2'); yyaxis right; stairs(r.hist.N); ylabel('N'); grid on
    if isfinite(allFires(m,1)), xline(allFires(m,1),'k--'); end
    title(sprintf('%dx%d',meshes(m,1),meshes(m,2))); xlabel('iteration');
end
exportgraphics(fig,fullfile(resultDir,'cross_resolution_gap_and_multiplicity.png'),'Resolution',180); close(fig)

% Plots 3--5: design metrics and why max step is misleading.
metricNames={'d_rms','moving','max_vs_rms'};
for p=1:3
    fig=figure('Visible','off','Color','w','Position',[100 100 1100 850]);
    tiledlayout(2,2,'TileSpacing','compact');
    for m=1:4
        r=allRuns{m}; nexttile
        if p==1
            semilogy(r.telemetry.d_rms); ylabel('RMS \Delta\rho');
        elseif p==2
            plot(r.telemetry.moving_fraction(1:4,:)'); ylabel('moving fraction');
        else
            semilogy(r.hist.dxOuter,'Color',[.15 .15 .15]); hold on
            semilogy(r.telemetry.d_rms,'Color',[.8 .2 0]); ylabel('design update');
        end
        grid on; if isfinite(allFires(m,1)), xline(allFires(m,1),'k--'); end
        title(sprintf('%dx%d',meshes(m,1),meshes(m,2))); xlabel('iteration');
    end
    if p==2, legend('>10^{-4}','>5x10^{-4}','>10^{-3}','>2.5x10^{-3}','Location','best'); end
    if p==3, legend('max |\Delta\rho|','RMS \Delta\rho','Location','best'); end
    exportgraphics(fig,fullfile(resultDir,['cross_resolution_' metricNames{p} '.png']),'Resolution',180); close(fig)
end

% Plot 6: 200-step post-fire objective evolution, normalized at fire.
fig=figure('Visible','off','Color','w','Position',[100 100 900 600]); hold on
for m=1:4
    r=allRuns{m}; k=allFires(m,1);
    if isfinite(k) && k+200<=r.nOuter
        ix=k:k+200; plot(0:200,r.hist.omega(1,ix)/mean(r.hist.omega(1,k-19:k))-1, ...
            'DisplayName',sprintf('%dx%d',meshes(m,1),meshes(m,2)));
    end
end
grid on; xlabel('iterations after shadow fire'); ylabel('relative \omega_1 change'); legend('Location','best');
exportgraphics(fig,fullfile(resultDir,'cross_resolution_delayed_postfire.png'),'Resolution',180); close(fig)

% Plot 7: locked candidates by mesh.
fig=figure('Visible','off','Color','w','Position',[100 100 850 500]);
bar(categorical(compose('%dx%d',meshes(:,1),meshes(:,2))),allFires); yline(1600,'k:');
ylabel('offline firing iteration'); legend('balanced','conservative','cap','Location','best'); grid on
exportgraphics(fig,fullfile(resultDir,'cross_resolution_firing_iterations.png'),'Resolution',180); close(fig)
disp(cross(:,{'mesh','role','detector','offline_fire','classification','objective_loss_abs', ...
    'topology_disagreement','pass_H50','pass_H100','pass_H200'}));
end

function row=localRow(r,mesh,role,d,k,rhoPhase,topoPhase)
row=struct(); row.mesh=sprintf('%dx%d',mesh(1),mesh(2)); row.role=role;
row.detector=d.name; row.offline_fire=k; row.full_iterations=r.nOuter;
if ~isfinite(k)
    row.iterations_saved=0; row.estimated_runtime_to_fire_seconds=NaN;
    row.full_runtime_seconds=r.wallclock; row.omega1=NaN; row.omega2=NaN; row.omega3=NaN;
    row.gap12_rel=NaN; row.N=NaN; row.step_rms=NaN; row.moving_fraction_gt_1e3=NaN;
    row.rho_phase_rms=NaN; row.topology_phase_turnover=NaN; row.objective_loss_signed=NaN;
    row.objective_loss_abs=NaN; row.density_L1=NaN; row.density_L2=NaN;
    row.topology_disagreement=NaN; row.connected_fire=false; row.connected_full=false;
    row.largest_component_fraction_fire=NaN; row.largest_component_fraction_full=NaN;
    row.volume=NaN; row.solver_healthy=false; row.bimodal=false;
    row.pass_H50=false; row.pass_H100=false; row.pass_H200=false;
    row.pass_obj_1pct=false; row.pass_obj_0p5pct=false; row.pass_obj_0p25pct=false; row.pass_obj_0p1pct=false;
    row.classification='NEVER_FIRES'; return
end
post=min(k+1,r.nOuter); R=double(r.telemetry.rho_snapshots);
rhoFire=R(:,k+1); rhoFull=double(r.rho); dw=r.hist.omega(1,post)-r.omega(1);
iterCost=r.hist.tEig+r.hist.tGrad+r.hist.tInner;
row.iterations_saved=r.nOuter-k;
row.estimated_runtime_to_fire_seconds=r.wallclock*sum(iterCost(1:k))/sum(iterCost);
row.full_runtime_seconds=r.wallclock;
row.omega1=r.hist.omega(1,post); row.omega2=r.hist.omega(2,post); row.omega3=r.hist.omega(3,post);
row.gap12_rel=abs(row.omega2-row.omega1)/row.omega1; row.N=r.hist.N(post);
row.step_rms=r.telemetry.d_rms(k); row.moving_fraction_gt_1e3=r.telemetry.moving_fraction(3,k);
row.rho_phase_rms=rhoPhase(k); row.topology_phase_turnover=topoPhase(k);
row.objective_loss_signed=-dw/r.omega(1); row.objective_loss_abs=abs(dw)/r.omega(1);
delta=rhoFire-rhoFull; row.density_L1=mean(abs(delta)); row.density_L2=sqrt(mean(delta.^2));
row.topology_disagreement=mean((rhoFire>=.5)~=(rhoFull>=.5));
[row.connected_fire,row.largest_component_fraction_fire]=localConnectivity(rhoFire,mesh(2),mesh(1));
[row.connected_full,row.largest_component_fraction_full]=localConnectivity(rhoFull,mesh(2),mesh(1));
row.volume=mean(rhoFire);
row.solver_healthy=all(r.hist.innerConv(1:k))&&all(r.telemetry.lp_flag(1:k)==1)&& ...
    all(r.telemetry.eig_ok(1:k))&&all(r.telemetry.finite_ok(1:k));
row.bimodal=row.N==2&&row.gap12_rel<=d.gap_tol;
row.pass_H50=localLabel(k,50,r); row.pass_H100=localLabel(k,100,r); row.pass_H200=localLabel(k,200,r);
row.pass_obj_1pct=row.objective_loss_abs<=.01; row.pass_obj_0p5pct=row.objective_loss_abs<=.005;
row.pass_obj_0p25pct=row.objective_loss_abs<=.0025; row.pass_obj_0p1pct=row.objective_loss_abs<=.001;
if k>1400, row.classification='TOO_LATE';
elseif row.pass_H50&&row.pass_H100&&row.pass_H200&&row.bimodal&&row.solver_healthy&&row.connected_fire
    row.classification='TRUE_POSITIVE';
else, row.classification='FALSE_POSITIVE';
end
end

function k=localFirstFire(r,d,rhoPhase,topoPhase)
n=r.nOuter; w=r.hist.omega(1,:); modeEvent=r.telemetry.mode_order_changed|r.telemetry.N_changed;
health=r.hist.innerConv&(r.telemetry.lp_flag==1)&r.telemetry.eig_ok&r.telemetry.finite_ok&~r.telemetry.eig_warning;
raw=false(1,n); fire=false(1,n); B=d.objective_block; W=d.window; MW=d.modal_window;
for q=max([2*B W+2 MW]):n
    newMean=mean(w(q-B+1:q)); oldMean=mean(w(q-2*B+1:q-B)); ix=q-W+1:q; im=q-MW+1:q;
    objPhase=abs(w(ix)-w(ix-2))./max(abs(w(ix)),eps);
    raw(q)=abs(newMean-oldMean)/max(abs(newMean),eps)<=d.objective_block_drift_tol && ...
        max(objPhase)<=d.objective_phase_recurrence_tol && max(rhoPhase(ix))<=d.rho_phase_rms_tol && ...
        max(topoPhase(ix))<=d.topology_phase_turnover_tol && all(r.hist.N(im)==d.required_N) && ...
        all(r.telemetry.gaps_rel(1,im)<=d.gap_tol) && ~any(modeEvent(im)) && all(health(im)) && ...
        abs(r.hist.vol(q)-r.cfg.volfrac)/r.cfg.volfrac<=d.volume_tol_rel;
    if q>=d.persistence, fire(q)=all(raw(q-d.persistence+1:q)); end
end
k=find(fire,1,'first'); if isempty(k), k=NaN; end
end

function tf=localLabel(k,H,r)
n=r.nOuter; q=k+H; if q>n, tf=false; return; end
if mod(q-k,2)==1, q=q-1; end
w=r.hist.omega(1,:); R=double(r.telemetry.rho_snapshots); B=20;
startMean=mean(w(k-B+1:k)); terminalMean=mean(w(end-B+1:end)); centres=k:B:q;
bm=NaN(size(centres)); for j=1:numel(centres), bm(j)=mean(w(centres(j)-B+1:centres(j))); end
topo=mean((R(:,k+1)>=.5)~=(R(:,q+1)>=.5)); future=k+1:q;
events=r.telemetry.mode_order_changed|r.telemetry.N_changed;
healthy=r.hist.innerConv&(r.telemetry.lp_flag==1)&r.telemetry.eig_ok&r.telemetry.finite_ok&~r.telemetry.eig_warning;
tf=abs(startMean-terminalMean)/terminalMean<=1e-3 && ...
    max(abs(bm-startMean))/terminalMean<=1e-3 && topo<=5e-3 && ...
    ~any(events(future))&&all(healthy(future))&&all(r.hist.N(future)==2)&& ...
    all(r.telemetry.gaps_rel(1,future)<=1e-2);
end

function [rhoPhase,topoPhase]=localPhaseMetrics(r)
n=r.nOuter;
if isfield(r.telemetry,'rho_phase_rms')
    rhoPhase=r.telemetry.rho_phase_rms; topoPhase=r.telemetry.topology_phase_turnover; return
end
R=double(r.telemetry.rho_snapshots); rhoPhase=NaN(1,n); topoPhase=NaN(1,n);
for k=2:n
    delta=R(:,k+1)-R(:,k-1); rhoPhase(k)=sqrt(mean(delta.^2));
    topoPhase(k)=mean((R(:,k+1)>=.5)~=(R(:,k-1)>=.5));
end
end

function [span,largestFraction]=localConnectivity(rho,nely,nelx)
A=reshape(rho>=.5,nely,nelx); seen=false(size(A)); largest=0; span=false;
for seed=find(A(:))'
    if seen(seed), continue; end
    queue=zeros(nnz(A),1); head=1; tail=1; queue(1)=seed; seen(seed)=true;
    count=0; touchesLeft=false; touchesRight=false;
    while head<=tail
        u=queue(head); head=head+1; count=count+1; [iy,ix]=ind2sub(size(A),u);
        touchesLeft=touchesLeft||(ix==1); touchesRight=touchesRight||(ix==nelx);
        nb=[iy-1 ix;iy+1 ix;iy ix-1;iy ix+1];
        for z=1:4
            y=nb(z,1); x=nb(z,2);
            if y>=1&&y<=nely&&x>=1&&x<=nelx&&A(y,x)&&~seen(y,x)
                tail=tail+1; queue(tail)=sub2ind(size(A),y,x); seen(y,x)=true;
            end
        end
    end
    largest=max(largest,count); span=span||(touchesLeft&&touchesRight);
end
largestFraction=largest/max(nnz(A),1);
end
