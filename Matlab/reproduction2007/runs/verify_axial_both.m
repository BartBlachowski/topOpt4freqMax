function verify_axial_both(meshes)
%VERIFY_AXIAL_BOTH  Initial design with ux restrained at BOTH ends (mid-height).
%   Tests the CLAUDE.md sec.3 candidate "axial restraint at both ends" against
%   the clean Fig. 4a read (iteration 1: omega ~ 71 / 245 / 428).
maxNumCompThreads(1); addpath('fem');
if nargin<1, meshes = {[160 20],[240 30]}; end
base = struct('a',8,'b',1,'t',1,'E',1e7,'nu',0.3,'rhom',1, ...
              'massType','consistent','axial','both','support','mid');
tg = struct('a',68.7,'b',104.1,'c',146.1);
for mi=1:numel(meshes)
  nelx=meshes{mi}(1); nely=meshes{mi}(2);
  fprintf('\n===== mesh %dx%d (NE=%d) =====\n',nelx,nely,nelx*nely);
  fprintf('%-3s %-5s %9s %8s %7s  %s\n','bc','elem','omega1','target','err %','omega2 omega3 omega4');
  for bc={'a','b','c'}
    for et={'Q4','Q6'}
      cfg=base; cfg.nelx=nelx; cfg.nely=nely; cfg.bc=bc{1}; cfg.elemType=et{1};
      mdl=model2D(cfg);
      [K,M]=assemble2D(mdl,0.5*ones(mdl.nele,1),3,'4');
      [w,Phi]=eigSolve(K,M,6,'eigs');
      T=classifyModes(mdl,M,Phi,w);
      ch=repmat(' ',1,3); for j=2:4, if T(j,2)>0.5, ch(j-1)='E'; else ch(j-1)='B'; end; end
      fprintf('%-3s %-5s %9.2f %8.1f %+7.2f  %7.1f%s %7.1f%s %7.1f%s\n', ...
        bc{1},et{1},w(1),tg.(bc{1}),100*(w(1)-tg.(bc{1}))/tg.(bc{1}), ...
        w(2),ch(1),w(3),ch(2),w(4),ch(3));
    end
  end
end
fprintf('\n(B = bending, E = extensional)\n');
end
