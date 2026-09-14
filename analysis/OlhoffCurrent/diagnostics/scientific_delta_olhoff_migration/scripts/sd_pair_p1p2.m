function sd_pair_p1p2()
P = sd_paths(); R = fullfile(P.snap,'repro','results');
A = load(fullfile(R,'P1_240_ped01_lin','res.mat')); B = load(fullfile(R,'P2_240_ped01_eq4','res.mat'));
D = sd_cfgdiff(A.cfg, B.cfg); fprintf('P1 vs P2: %s\n', strjoin({D.path}, ', '));
for L = {A, B}
    r = L{1}.res; w = r.hist.omega(1,:);
    fprintf('%s status %s outer %d spikes %d Mnd %.4f\n', r.cfg.runtime.name, r.status, r.nOuter, sum(w(2:end) < 0.7*w(1:end-1)), 4*mean(r.rho.*(1-r.rho)));
end
end
