function [cfg,policy]=olhoff_cfg(nelx,nely,horizon)
%OLHOFF_CFG Frozen S1 Olhoff profile with only mesh and horizon reduced.
[cfg,~]=repro2007_config('fig3a_best');
cfg.nelx=nelx; cfg.nely=nely; cfg.maxOuter=horizon; cfg.threads=1; cfg.verbose=false;
policy=struct('id','S1','move_sequence',[.005 .0025],'gap_threshold',.01,'persistence',100);
end
