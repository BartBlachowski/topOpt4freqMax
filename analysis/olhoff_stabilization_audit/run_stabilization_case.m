function outFile=run_stabilization_case(profileId,nelx,nely,outDir,maxOuter)
%RUN_STABILIZATION_CASE Run one preregistered stabilization case.
if nargin<5,maxOuter=1600;end
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(fullfile(repo,'Matlab','reproduction2007','runner'));addpath(here);
guard=repro2007_paths(); %#ok<NASGU>
[cfg,~]=repro2007_config('fig3a_best');cfg.nelx=nelx;cfg.nely=nely;
cfg.maxOuter=maxOuter;cfg.verbose=false;cfg.threads=1;
switch upper(profileId)
    case 'S0',seq=0.005;
    case 'S1',seq=[0.005 0.0025];
    case 'S2',seq=[0.005 0.0025 0.00125];
    case 'S3',seq=[0.005 0.0025 0.00125 0.000625];
    otherwise,error('Unknown preregistered profile %s',profileId)
end
policy=struct('id',upper(profileId),'move_sequence',seq,'gap_threshold',0.01,'persistence',100);
fprintf('Starting %s %dx%d cap=%d\n',profileId,nelx,nely,maxOuter);
res=olhoffOptStabilized(cfg,policy);
if exist(outDir,'dir')~=7,mkdir(outDir);end
outFile=fullfile(outDir,sprintf('%s_%dx%d.mat',lower(profileId),nelx,nely));
save(outFile,'res','-v7.3');
fprintf('Saved %s status=%s n=%d triggers=%s wall=%.3fs\n',outFile,res.status,res.nOuter,mat2str(res.trigger_iterations),res.wallclock);
end
