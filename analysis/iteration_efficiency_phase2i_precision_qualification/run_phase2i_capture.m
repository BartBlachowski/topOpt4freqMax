% Phase 2I WP2/WP3: same-state capture and determinism evidence.
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
outDir=fileparts(mfilename('fullpath')); rawDir=fullfile(outDir,'raw');
if ~isfolder(rawDir),mkdir(rawDir);end
addpath(outDir,fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
    fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck'), ...
    fullfile(repo,'analysis','olhoff_stabilization_audit'), ...
    fullfile(repo,'Matlab','reproduction2007','runner'));
guard=repro2007_paths(); %#ok<NASGU>
maxNumCompThreads(1);

nelx=96;nely=12;H=3200;
[cfg,policy]=ie2br.olhoff_cfg(nelx,nely,H);
fprintf('Phase 2I double capture %dx%d H=%d\n',nelx,nely,H);
t=tic; rd=olhoffOptStabilizedDoubleCapture(cfg,policy);captureWall=toc(t);
assert(rd.nOuter==H&&strcmp(rd.status,'CAP_HIT'),'Double capture did not reach B_ref.');
assert(isa(rd.rho,'double')&&isa(rd.rho_snapshots,'double'));
Xd=rd.rho_snapshots; assert(isequal(rd.rho,Xd(:,end)));

fprintf('Phase 2I protected full repeat %dx%d H=%d\n',nelx,nely,H);
t=tic; rp=olhoffOptStabilized(cfg,policy);protectedWall=toc(t);
assert(rp.nOuter==H&&strcmp(rp.status,'CAP_HIT'),'Protected repeat did not reach B_ref.');
assert(isa(rp.rho,'double')&&isa(rp.rho_snapshots,'single'));
assert(isequal(rp.rho,rd.rho),'Final double state differs between capture and protected runner.');
assert(isequal(rp.rho_snapshots,single(Xd)),'Double-capture cast differs from protected snapshots.');
assert(localHistoryEqual(rp.hist,rd.hist),'Non-timing history differs between capture and protected runner.');
assert(isequal(rp.trigger_iterations,rd.trigger_iterations));
Xs=rp.rho_snapshots;

% Strategically selected capped prefixes exercise the early trajectory,
% rho=.1 parking, mid trajectory, all three primary-q endpoint regions,
% reference establishment, and the late/reference cap.  Each capped run is
% compared with the lossless full-run column, not merely with a float32 image.
prefixK=[80 100 252 400 453 552 2100 3200];
prefixRole=["rho_0p1_parking";"early";"difficult_anchor";"mid"; ...
    "near_q_0p995_enter";"near_q_0p995_cert";"reference_establishment";"late_B_ref"];
nr=numel(prefixK);densityId=false(nr,1);singleId=false(nr,1);castId=false(nr,1);
historyId=false(nr,1);outerId=false(nr,1);status=strings(nr,1);prefixWall=zeros(nr,1);
for ii=1:nr
    kPrefix=prefixK(ii);cp=cfg;cp.maxOuter=kPrefix;
    t=tic;rr=olhoffOptStabilized(cp,policy);prefixWall(ii)=toc(t);
    outerId(ii)=rr.nOuter==kPrefix;status(ii)=string(rr.status);
    densityId(ii)=isequal(rr.rho,Xd(:,kPrefix+1));
    singleId(ii)=isequal(rr.rho_snapshots(:,end),Xs(:,kPrefix+1));
    castId(ii)=isequal(single(rr.rho),rr.rho_snapshots(:,end));
    historyId(ii)=localHistoryPrefix(rr.hist,rp.hist,kPrefix);
    assert(outerId(ii)&&strcmp(rr.status,'CAP_HIT')&&densityId(ii)&&singleId(ii)&&castId(ii)&&historyId(ii));
end
T=table(prefixK(:),prefixRole,densityId,singleId,castId,historyId,outerId,status,prefixWall, ...
    'VariableNames',{'k','trajectory_role','double_density_identical','single_snapshot_identical', ...
    'cast_identity','objective_frequency_history_identical','outer_count_identical','native_status','wall_seconds'});
writetable(T,fullfile(outDir,'PREFIX_DETERMINISM.csv'));

rdScientific=rmfield(rd,{'mdl'});rdScientific.hist=rmfield(rdScientific.hist,{'tEig','tGrad','tInner'}); %#ok<NASGU>
rpScientific=rmfield(rp,{'mdl'});rpScientific.hist=rmfield(rpScientific.hist,{'tEig','tGrad','tInner'}); %#ok<NASGU>
save(fullfile(rawDir,'capture_96x12_H3200.mat'),'Xd','Xs','cfg','policy', ...
    'rdScientific','rpScientific','captureWall','protectedWall','prefixWall','-v7.3');
fprintf('CAPTURE_PASS capture=%.1fs protected=%.1fs prefixes_total=%.1fs\n', ...
    captureWall,protectedWall,sum(prefixWall));

function ok=localHistoryEqual(a,b)
fields=setdiff(fieldnames(a),{'tEig','tGrad','tInner'});ok=true;
for i=1:numel(fields),ok=ok&&isequaln(a.(fields{i}),b.(fields{i}));end
end
function ok=localHistoryPrefix(a,b,k)
fields=setdiff(fieldnames(a),{'tEig','tGrad','tInner'});ok=true;
for i=1:numel(fields)
    av=a.(fields{i});bv=b.(fields{i});
    if size(av,2)==k&&size(bv,2)>=k,bv=bv(:,1:k);
    else,bv=bv(1:numel(av));bv=reshape(bv,size(av));end
    ok=ok&&isequaln(av,bv);
end
end
