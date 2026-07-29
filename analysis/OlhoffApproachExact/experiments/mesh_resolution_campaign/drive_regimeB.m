meshes = [40 5; 80 10; 160 20; 240 30];
bcs = {'CC','SS','CS'};
for b = 1:numel(bcs)
    for m = 1:size(meshes,1)
        fprintf('\n>>> %s %dx%d\n', bcs{b}, meshes(m,1), meshes(m,2));
        run_mesh_campaign(bcs{b}, 'B', meshes(m,1), meshes(m,2), 0);
    end
end
