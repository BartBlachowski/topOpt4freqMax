function sweep_supports()
%SWEEP_SUPPORTS  Break the support/element/mesh confound of CLAUDE.md sec.2
%   using BOTH discriminators: omega_1 (68.7) and the extensional mode that
%   Fig. 4a places at ~428 at iteration 1.
maxNumCompThreads(1);
base = struct('a',8,'b',1,'t',1,'E',1e7,'nu',0.3,'rhom',1, ...
              'massType','consistent','bc','a','elemType','Q4');
meshes = {[64 8],[80 10],[160 20],[240 30]};
fprintf('%-8s %-7s %-6s %9s %8s   %9s %9s %9s\n', ...
        'mesh','support','axial','omega1','err %','omega2','omega3','omega4');
for mi=1:numel(meshes)
  for supp={'mid','corner'}
    for ax={'one','both'}
      cfg=base; cfg.nelx=meshes{mi}(1); cfg.nely=meshes{mi}(2);
      cfg.support=supp{1}; cfg.axial=ax{1};
      mdl=model2D(cfg);
      [K,M]=assemble2D(mdl,0.5*ones(mdl.nele,1),3,'4');
      [w,Phi]=eigSolve(K,M,5,'eigs');
      T=classifyModes(mdl,M,Phi,w);
      tag=repmat(' ',1,3);
      for j=2:4, if T(j,2)>0.5, tag(j-1)='E'; else, tag(j-1)='B'; end; end
      fprintf('%-8s %-7s %-6s %9.2f %+8.2f   %8.1f%s %8.1f%s %8.1f%s\n', ...
        sprintf('%dx%d',cfg.nelx,cfg.nely),supp{1},ax{1},w(1), ...
        100*(w(1)-68.7)/68.7,w(2),tag(1),w(3),tag(2),w(4),tag(3));
    end
  end
end
fprintf('\nPaper: omega1 = 68.7 ; Fig.4a iteration 1 reads ~71 / ~245 / ~428\n');
fprintf('(B = bending, E = extensional)\n');
end
