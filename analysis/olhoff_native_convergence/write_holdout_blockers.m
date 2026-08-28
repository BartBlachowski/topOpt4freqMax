function write_holdout_blockers()
%WRITE_HOLDOUT_BLOCKERS Explain locked-detector non-firing without retuning.
here=fileparts(mfilename('fullpath')); resultDir=fullfile(here,'results');
j=jsondecode(fileread(fullfile(resultDir,'native_convergence_config.json'))); d=j.selected_detector;
meshes=[160 20;240 30;320 40;400 50]; out=cell(4,1);
for m=1:4
    s=load(fullfile(resultDir,sprintf('development_%dx%d.mat',meshes(m,1),meshes(m,2))),'res'); r=s.res;
    [rp,tp]=localPhase(r); n=r.nOuter; w=r.hist.omega(1,:);
    ev=r.telemetry.mode_order_changed|r.telemetry.N_changed;
    healthy=r.hist.innerConv&(r.telemetry.lp_flag==1)&r.telemetry.eig_ok&r.telemetry.finite_ok&~r.telemetry.eig_warning;
    B=d.objective_block; W=d.window; MW=d.modal_window; first=max([2*B W+2 MW]);
    objBlock=false(1,n); objPhase=false(1,n); designRms=false(1,n); topo=false(1,n);
    modal=false(1,n); health=false(1,n); raw=false(1,n);
    for k=first:n
        nm=mean(w(k-B+1:k)); om=mean(w(k-2*B+1:k-B)); ix=k-W+1:k; im=k-MW+1:k;
        objBlock(k)=abs(nm-om)/nm<=d.objective_block_drift_tol;
        objPhase(k)=max(abs(w(ix)-w(ix-2))./w(ix))<=d.objective_phase_recurrence_tol;
        designRms(k)=max(rp(ix))<=d.rho_phase_rms_tol;
        topo(k)=max(tp(ix))<=d.topology_phase_turnover_tol;
        modal(k)=all(r.hist.N(im)==d.required_N)&&all(r.telemetry.gaps_rel(1,im)<=d.gap_tol)&&~any(ev(im));
        health(k)=all(healthy(im))&&abs(r.hist.vol(k)-r.cfg.volfrac)/r.cfg.volfrac<=d.volume_tol_rel;
        raw(k)=objBlock(k)&&objPhase(k)&&designRms(k)&&topo(k)&&modal(k)&&health(k);
    end
    q=first:n; z=struct(); z.mesh=sprintf('%dx%d',meshes(m,1),meshes(m,2));
    z.objective_block_pass_fraction=mean(objBlock(q)); z.objective_phase_pass_fraction=mean(objPhase(q));
    z.design_phase_rms_pass_fraction=mean(designRms(q)); z.topology_phase_pass_fraction=mean(topo(q));
    z.modal_guard_pass_fraction=mean(modal(q)); z.health_feasibility_pass_fraction=mean(health(q));
    z.raw_all_pass_count=nnz(raw); z.longest_raw_all_run=localLongest(raw);
    z.mode_event_count=nnz(ev); last=find(ev,1,'last'); if isempty(last),last=NaN;end; z.last_mode_event=last;
    z.final_omega1=r.omega(1); z.final_omega2=r.omega(2); z.final_omega3=r.omega(3);
    z.final_gap12_rel=abs(r.omega(2)-r.omega(1))/r.omega(1); z.final_N=r.hist.N(end);
    z.last200_max_objective_phase=max(abs(w(end-199:end)-w(end-201:end-2))./w(end-199:end));
    z.last200_max_rho_phase=max(rp(end-199:end)); z.last200_max_topology_phase=max(tp(end-199:end));
    z.last200_max_gap12=max(r.telemetry.gaps_rel(1,end-199:end));
    z.solver_failure_count=nnz(~healthy); z.final_volume_relative_error=abs(mean(r.rho)-r.cfg.volfrac)/r.cfg.volfrac;
    [z.final_spans_supports,z.final_largest_component_fraction]=localConnectivity(r.rho,meshes(m,2),meshes(m,1));
    out{m}=z;
end
T=struct2table(vertcat(out{:}));
writetable(T,fullfile(resultDir,'native_convergence_holdout_blockers.csv')); disp(T);
end

function longest=localLongest(x)
d=diff([false x false]); starts=find(d==1); stops=find(d==-1)-1;
if isempty(starts), longest=0; else, longest=max(stops-starts+1); end
end

function [rp,tp]=localPhase(r)
n=r.nOuter;
if isfield(r.telemetry,'rho_phase_rms'), rp=r.telemetry.rho_phase_rms;tp=r.telemetry.topology_phase_turnover;return;end
R=double(r.telemetry.rho_snapshots);rp=NaN(1,n);tp=NaN(1,n);
for k=2:n,delta=R(:,k+1)-R(:,k-1);rp(k)=sqrt(mean(delta.^2));tp(k)=mean((R(:,k+1)>=.5)~=(R(:,k-1)>=.5));end
end

function [span,largestFraction]=localConnectivity(rho,nely,nelx)
A=reshape(rho>=.5,nely,nelx);seen=false(size(A));largest=0;span=false;capacity=nnz(A);
for seed=find(A(:))'
    if seen(seed),continue;end
    queue=zeros(capacity,1);head=1;tail=1;queue(1)=seed;seen(seed)=true;count=0;L=false;R=false;
    while head<=tail
        u=queue(head);head=head+1;count=count+1;[y,x]=ind2sub(size(A),u);L=L||x==1;R=R||x==nelx;
        nb=[y-1 x;y+1 x;y x-1;y x+1];
        for z=1:4
            yy=nb(z,1);xx=nb(z,2);
            if yy>=1&&yy<=nely&&xx>=1&&xx<=nelx&&A(yy,xx)&&~seen(yy,xx)
                tail=tail+1;queue(tail)=sub2ind(size(A),yy,xx);seen(yy,xx)=true;
            end
        end
    end
    largest=max(largest,count);span=span||(L&&R);
end
largestFraction=largest/max(capacity,1);
end
