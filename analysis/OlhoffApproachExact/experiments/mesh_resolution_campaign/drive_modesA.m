meshes = [40 5; 80 10; 160 20; 240 30];
for m = 1:size(meshes,1)
    postprocess_modes(sprintf('CC_regimeA_%dx%d_s0', meshes(m,1), meshes(m,2)));
end
