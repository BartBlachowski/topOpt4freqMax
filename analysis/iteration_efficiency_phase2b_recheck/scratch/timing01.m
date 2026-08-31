repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','olhoff_stabilization_audit'),fullfile(repo,'Matlab','reproduction2007','runner'));
guard=repro2007_paths(); %#ok<NASGU>
policy=struct('id','S1','move_sequence',[.005 .0025],'gap_threshold',.01,'persistence',100);
for m=[48 6; 80 10; 96 12]'
  [cfg,~]=repro2007_config('fig3a_best'); cfg.nelx=m(1); cfg.nely=m(2); cfg.threads=1; cfg.verbose=false;
  cfg.maxOuter=50; t=tic; r=olhoffOptStabilized(cfg,policy); el=toc(t);
  fprintf('%dx%d  nele=%d  50 iters in %.3f s  -> %.2f ms/iter  status=%s rho0=%.6g\n', ...
     m(1),m(2),cfg.nelx*cfg.nely,el,1000*el/50,r.status,cfg.rho0);
end
