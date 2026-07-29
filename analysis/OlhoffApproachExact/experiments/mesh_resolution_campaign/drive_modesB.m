bcs = {'CC','SS','CS'}; meshes = [40 5; 80 10; 160 20; 240 30];
for b = 1:numel(bcs)
    for m = 1:size(meshes,1)
        postprocess_modes(sprintf('%s_regimeB_%dx%d_s0', bcs{b}, meshes(m,1), meshes(m,2)));
    end
end
