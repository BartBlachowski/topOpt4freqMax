function probe_timing()
addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'Matlab'));
rc=struct('verbose',true,'formulation','olhoff','optimizer','mma', ...
    'max_outer_iterations',3,'max_inner_iterations',500,'persistence',1000);
t=tic;[~,w,info]=topopt_olhoff_regularized(160,20,.5,3,1.3,.005,'simply',rc);
fprintf('TIMING outer=%d wall=%.2fs per-outer=%.2fs inner=%d w1=%.6f status=%s\n', ...
    info.nOuter,toc(t),toc(t)/info.nOuter,info.iterations.inner_total,w(1),info.status);
end
