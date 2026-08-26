function diag_modes(nelx, nely, axialOpts)
%DIAG_MODES  Identify the character of the low modes of the initial design,
%   case (a).  This is the discriminator for CLAUDE.md sec.3.
maxNumCompThreads(1); addpath('fem');
if nargin<3, axialOpts = {'one','both'}; end
base = struct('a',8,'b',1,'t',1,'E',1e7,'nu',0.3,'rhom',1, ...
              'massType','consistent','bc','a','elemType','Q4','support','mid', ...
              'nelx',nelx,'nely',nely);
for ax = axialOpts
  for supp = {'mid','face'}
    cfg = base; cfg.axial = ax{1}; cfg.support = supp{1};
    mdl = model2D(cfg);
    [K,M] = assemble2D(mdl, 0.5*ones(mdl.nele,1), 3, '4');
    [w,Phi] = eigSolve(K,M,6,'eigs');
    T = classifyModes(mdl,M,Phi,w);
    fprintf('\n--- case a, %dx%d, support=%s, axial=%s ---\n',nelx,nely,supp{1},ax{1});
    fprintf('  j  omega     Ex      Ey    signchg  character\n');
    for j=1:6
        if T(j,2)>0.5, ch='EXTENSIONAL'; else, ch=sprintf('bending n=%d',max(T(j,4)-1,0)); end
        fprintf('%3d %8.2f %7.3f %7.3f %6d   %s\n',j,T(j,1),T(j,2),T(j,3),T(j,4),ch);
    end
  end
end
end
