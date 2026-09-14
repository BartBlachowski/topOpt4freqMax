function sd_retained_pairs()
%SD_RETAINED_PAIRS  Leaf-by-leaf admission check for the retained material-factor pairs.
P = sd_paths(); R = fullfile(P.snap,'repro','results');
A = load(fullfile(R,'A2_240_d010','res.mat')); S = load(fullfile(R,'S240x30','res.mat'));
D = sd_cfgdiff(A.cfg, S.cfg);
fprintf('A2 vs S240 differing leaves:\n'); for i=1:numel(D), fprintf('  %s\n', D(i).path); end
h = A.res.hist; w = h.omega(1,:); fprintf('A2: status %s outer %d spikes %d Mnd %.4f | S240 spikes %d\n', A.res.status, A.res.nOuter, sum(w(2:end)<0.7*w(1:end-1)), 4*mean(A.res.rho.*(1-A.res.rho)), sum(S.res.hist.omega(1,2:end)<0.7*S.res.hist.omega(1,1:end-1)));
out.A2_vs_S240 = {D.path};
fid=fopen(fullfile(P.eval,'retained_pairs.json'),'w'); fprintf(fid,'%s\n',jsonencode(out,'PrettyPrint',true)); fclose(fid);
end
