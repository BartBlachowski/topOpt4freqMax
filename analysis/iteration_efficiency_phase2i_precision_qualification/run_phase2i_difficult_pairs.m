% Phase 2I WP7: obtain same-state pairs for known Candidate-C difficult cases.
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
outDir=fileparts(mfilename('fullpath'));rawDir=fullfile(outDir,'raw');
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck'), ...
    fullfile(repo,'analysis','olhoff_stabilization_audit'), ...
    fullfile(repo,'Matlab','reproduction2007','runner'));
guard=repro2007_paths(); %#ok<NASGU>
maxNumCompThreads(1);
meshes=[160 20;240 30];ks=[252 594];labels=["k252_anchor";"k594_disagreement"];
pairs=struct([]);rows={};
for i=1:size(meshes,1)
    nx=meshes(i,1);ny=meshes(i,2);k=ks(i);
    [cfg,policy]=ie2br.olhoff_cfg(nx,ny,k);
    fprintf('Phase 2I difficult pair %dx%d k=%d\n',nx,ny,k);
    t=tic;r=olhoffOptStabilized(cfg,policy);wall=toc(t);
    assert(r.nOuter==k&&strcmp(r.status,'CAP_HIT'));
    baselineFile=fullfile(repo,'examples','Performance','final_campaign','raw','olhoff',sprintf('s1_%dx%d.mat',nx,ny));
    S=load(baselineFile,'res');
    baselineCompatible=size(S.res.rho_snapshots,2)>=k+1;
    if baselineCompatible
        prefixIdentical=isequal(r.rho_snapshots(:,end),S.res.rho_snapshots(:,k+1));
    else,prefixIdentical=false;end
    castIdentity=isequal(single(r.rho),r.rho_snapshots(:,end));
    assert(prefixIdentical&&castIdentity,'Difficult-case prefix/cast identity failed.');
    pairs(i).label=labels(i);pairs(i).nelx=nx;pairs(i).nely=ny;pairs(i).k=k;
    pairs(i).x_double=r.rho;pairs(i).x_single=r.rho_snapshots(:,end);
    pairs(i).prefix_identical=prefixIdentical;pairs(i).cast_identity=castIdentity;
    pairs(i).native_status=r.status;pairs(i).wall_seconds=wall;
    rows(i,:)={labels(i),sprintf('%dx%d',nx,ny),k,prefixIdentical,castIdentity,string(r.status),wall}; %#ok<AGROW>
end
writetable(cell2table(rows,'VariableNames',{'case_id','mesh','k','prefix_identical','cast_identity','native_status','wall_seconds'}), ...
    fullfile(outDir,'DIFFICULT_CASE_PREFIX_IDENTITY.csv'));
save(fullfile(rawDir,'difficult_pairs.mat'),'pairs','-v7.3');
