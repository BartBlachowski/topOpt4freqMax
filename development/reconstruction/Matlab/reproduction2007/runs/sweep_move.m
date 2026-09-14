function sweep_move(moves, tolMults, nOuter, maxInner)
%SWEEP_MOVE  The move limit and the multiplicity tolerance are BOTH unstated
%   in the paper (CLAUDE.md sec.4).  Sweep them and report what each does to
%   the optimum and to the coalescence that Fig. 4a shows.
if nargin<1, moves    = [0.1 0.05 0.02 0.01 0.005]; end
if nargin<2, tolMults = 0.02; end
if nargin<3, nOuter   = 120; end
if nargin<4, maxInner = 30;  end
fprintf('%-8s %-8s %8s %8s %8s %6s %8s %8s\n', ...
        'move','tolMult','omega1','omega2','omega3','Nend','coal%','maxw1');
for mv = moves
  for tm = tolMults
    cfg = defaultCfg();
    cfg.move = mv; cfg.tolMult = tm; cfg.maxOuter = nOuter;
    cfg.maxInner = maxInner; cfg.verbose = false;
    res = olhoffOpt(cfg);
    h = res.hist;
    last = max(1, numel(h.N)-19):numel(h.N);
    coal = 100*mean(h.N(last) >= 2);
    fprintf('%-8.3f %-8.3f %8.2f %8.2f %8.2f %6d %8.0f %8.2f\n', ...
        mv, tm, res.omega(1), res.omega(2), res.omega(3), h.N(end), coal, max(h.omega(1,:)));
    save(sprintf('results/sweep_mv%g_tm%g.mat',mv,tm),'res','-v7.3');
  end
end
fprintf('\nTarget: omega1_opt = 174.7, BIMODAL (coal%% should be 100)\n');
end
