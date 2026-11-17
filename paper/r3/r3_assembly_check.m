function r3_assembly_check(oldDir)
%R3_ASSEMBLY_CHECK  Compare the lower-triangle assemble_KM with the previous 64-triplet
%   version (assemble_KM_old.m in oldDir) on 160x20, 400x50, 800x100: max abs and relative
%   difference of K and M (random design, SIMP and Pedersen law), and time per K+M assembly.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
addpath(oldDir);  maxNumCompThreads(1);
meshes = [160 20; 400 50; 800 100];
for k = 1:3
    prob = problem_ss_beam(meshes(k,1), meshes(k,2));  mdl = fe_mesh(prob);
    rng(1);  x = rand(mdl.nEl, 1);
    for xLin = [0 0.1]
        [K1, M1, E1, r1] = assemble_KM_old(mdl, x, prob.E, prob.Emin, prob.rho, prob.rhoMin, 3, xLin);
        [K2, M2, E2, r2] = assemble_KM(mdl, x, prob.E, prob.Emin, prob.rho, prob.rhoMin, 3, xLin);
        dK = max(abs(K1(:) - K2(:)));  dM = max(abs(M1(:) - M2(:)));
        fprintf('%4dx%-3d xLin=%.1f  max|dK| = %.3e (rel %.3e)  max|dM| = %.3e (rel %.3e)  nnz %d/%d  Ee,re equal %d\n', ...
            meshes(k,:), xLin, full(dK), full(dK/max(abs(K1(:)))), full(dM), full(dM/max(abs(M1(:)))), nnz(K1), nnz(K2), ...
            isequal(E1, E2) && isequal(r1, r2));
    end
    n = 5;  t0 = tic;  for i = 1:n, assemble_KM_old(mdl, x, prob.E, prob.Emin, prob.rho, prob.rhoMin); end;  to = toc(t0)/n;
    t0 = tic;  for i = 1:n, assemble_KM(mdl, x, prob.E, prob.Emin, prob.rho, prob.rhoMin); end;  tn = toc(t0)/n;
    fprintf('%4dx%-3d time per K+M assembly: old %.3f s, new %.3f s\n', meshes(k,:), to, tn);
end
end
