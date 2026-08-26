function verify_initial(meshes)
%VERIFY_INITIAL  Initial-design eigenfrequencies of the 2D beam, Figs. 2(a-c).
%
%   Sweeps the two idealizations that CLAUDE.md sec.2 flags as CONFOUNDED --
%   support placement and element formulation -- so the confounding can be
%   broken rather than assumed away.  Paper targets: 68.7 / 104.1 / 146.1.

if nargin < 1, meshes = {[160 20]}; end
maxNumCompThreads(1);
addpath('fem');

base = struct('a',8,'b',1,'t',1,'E',1e7,'nu',0.3,'rhom',1, ...
              'massType','consistent','axial','one');
p = 3;  massInterp = '4';  rho0 = 0.5;

targets = struct('a',68.7,'b',104.1,'c',146.1);

for mi = 1:numel(meshes)
    nelx = meshes{mi}(1); nely = meshes{mi}(2);
    fprintf('\n================ mesh %d x %d  (NE = %d) ================\n', ...
            nelx, nely, nelx*nely);
    fprintf('%-4s %-7s %-7s %10s %10s %8s   %s\n', ...
            'bc','elem','supp','omega1','target','err %','omega_2..4');
    for bc = {'a','b','c'}
      for elemType = {'Q4','Q6'}
        for supp = {'mid','corner','face'}
            cfg = base;
            cfg.nelx = nelx; cfg.nely = nely; cfg.bc = bc{1};
            cfg.elemType = elemType{1}; cfg.support = supp{1};
            if strcmp(bc{1},'c') && ~strcmp(supp{1},'mid')
                continue     % clamped-clamped has no simple support to place
            end
            mdl = model2D(cfg);
            rho = rho0*ones(mdl.nele,1);
            [K,M] = assemble2D(mdl, rho, p, massInterp);
            w = eigSolve(K, M, 6, 'eigs');
            tgt = targets.(bc{1});
            fprintf('%-4s %-7s %-7s %10.2f %10.1f %+8.2f   %7.1f %7.1f %7.1f\n', ...
                    bc{1}, elemType{1}, supp{1}, w(1), tgt, ...
                    100*(w(1)-tgt)/tgt, w(2), w(3), w(4));
        end
      end
    end
end
end
