repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck'));
[p,guard]=ie2br.setup_paths(); %#ok<ASGLU>
out=p.phase2br; s01=single(0.1);
meshes={'160x20','240x30','320x40','400x50','480x60','560x70','640x80','720x90','800x100'};
nx=[160 240 320 400 480 560 640 720 800]; ny=[20 30 40 50 60 70 80 90 100];
inv={}; risk={};
for i=1:9
  f=fullfile(repo,'examples','Performance','final_campaign','raw','olhoff',['s1_' meshes{i} '.mat']);
  d=dir(f);
  if d.bytes==0
    inv(i,:)={meshes{i},d.bytes,0,'N/A','N/A','N/A','N/A',false,false,'RUN_ERROR / N/A / UNVERIFIABLE_AT_PRESENT'}; %#ok<AGROW>
    risk(i,:)={meshes{i},0,0,0,0,0,0,0.01*nx(i)*ny(i)/8,'UNAVAILABLE'}; %#ok<AGROW>
    continue
  end
  S=load(f,'res'); r=S.res; X=r.rho_snapshots; [ne,ns]=size(X);
  % Patch 8: verify endpoint semantics per run rather than assuming
  okDims = ismatrix(X) && ns==r.nOuter+1 && ne==numel(r.rho);
  okCast = isequal(X(:,end),single(r.rho));
  inv(i,:)={meshes{i},d.bytes,ns,class(X),sprintf('%dx%d',ne,ns),class(r.rho),r.status,okDims,okCast, ...
     'AVAILABLE'}; %#ok<AGROW>
  hit=(X==s01); per=sum(hit,1);
  nsl=round(0.5*ne); tie=false(1,ns-1);
  for k=2:ns
    x=double(X(:,k)); [~,o]=sortrows([-x,(1:ne).'],[1 2]);
    tie(k-1)= X(o(nsl),k)==X(o(nsl+1),k);
  end
  risk(i,:)={meshes{i},ns,ne,sum(per),nnz(per>0),max(per),nnz(tie),0.01*nx(i)*ny(i)/8,'AVAILABLE'}; %#ok<AGROW>
  fprintf('%s  states=%d atrisk_states=%d maxper=%d ties=%d dims_ok=%d cast_ok=%d status=%s\n', ...
     meshes{i},ns,nnz(per>0),max(per),nnz(tie),okDims,okCast,r.status);
  clear S r X hit
end
writetable(cell2table(inv,'VariableNames',{'mesh','file_bytes','stored_states','snapshot_class','snapshot_dims', ...
 'returned_final_class','run_status','dims_and_count_verified','endpoint_cast_verified','availability'}), ...
 fullfile(out,'PAIRED_EVIDENCE_INVENTORY.csv'));
writetable(cell2table(risk,'VariableNames',{'mesh','stored_states','n_elements','atrisk_elements_total', ...
 'atrisk_states','max_atrisk_per_state','cutoff_tie_states','a_sig_elements','availability'}), ...
 fullfile(out,'PRODUCTION_SCALE_RISK.csv'));

%% tie-collapse mechanism verification on the one differing sampled pair
Rr=load(fullfile(p.runs,'resolve_96x12.mat'));
j=find(Rr.binDiff>0);
fprintf('\nsampled genuine pairs with binary difference: %d\n',numel(j));
if ~isempty(j)
  [cfg,policy]=ie2br.olhoff_cfg(96,12,3200);
  for t=1:numel(j)
    k=Rr.sample(j(t)); c=cfg; c.maxOuter=k; r=olhoffOptStabilized(c,policy);
    xd=double(r.rho); xs=double(r.rho_snapshots(:,end)); ne=numel(xd); nsl=round(0.5*ne);
    bd=ie2a.exact_count_binary(xd,0.5); bs=ie2a.exact_count_binary(xs,0.5);
    dif=find(bd~=bs); [~,od]=sortrows([-xd,(1:ne).'],[1 2]);
    sv=single(r.rho); tieval=sv(od(nsl));
    grp=find(sv==tieval);
    fprintf('k=%d differing elements=%d ; single tie-group size at cutoff=%d ; cutoff rank=%d\n',k,numel(dif),numel(grp),nsl);
    fprintf('   all differing elements inside the single tie group: %d\n',all(ismember(dif,grp)));
    fprintf('   double gap across cutoff = %.3e (one float32 ulp at 0.1 = %.3e)\n', ...
        xd(od(nsl))-xd(od(nsl+1)), double(eps(single(0.1))));
  end
end
