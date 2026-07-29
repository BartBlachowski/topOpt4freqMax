% Regime control: the committed run_clamped_clamped_exact.m configuration,
% verbatim, across the same mesh ladder.  Only nelx,nely change.
meshes = [40 5; 80 10; 160 20; 240 30];
for m = 1:size(meshes,1)
    run_mesh_campaign('CC', 'A', meshes(m,1), meshes(m,2), 0);
end
