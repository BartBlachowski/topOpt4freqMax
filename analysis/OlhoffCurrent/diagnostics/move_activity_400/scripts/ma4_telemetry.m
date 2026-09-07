function P = ma4_telemetry(h, RHO, DRHO, NE)
%MA4_TELEMETRY  Per-iteration record incl. the FULL element-activity distribution.
%
%   Everything is POST-HOC: computed from the recorded trajectory, nothing fed
%   back into the solver.
%
%   UNITS, stated once and precisely (preregistration sec. 6):
%
%     u_e(k) = |rho_e(k) - rho_e(k-1)| / move(k)          DIMENSIONLESS
%
%     move(k) is the move that BOUNDED the increment stored at index k, so it
%     governs the transition rho(k-1) -> rho(k).  Established by reading
%     olhoffSolve.m (mvNow computed at l.267, passed as the box at l.298, stored
%     at l.381 beside the increment statistics at l.373/380), NOT assumed.
%
%     Active-set thresholds tau are on the RAW increment |Delta rho_e|, NOT on
%     u.  This is move_stop's convention (ms_run.m:77) and is preserved so that
%     160x20, 320x40 and 400x50 counts are directly comparable.
%       tau = [1e-4, epsRMS, 1e-3, 1e-2],  epsRMS = stop.tolerance/sqrt(NE)
%     which is 8.83883476483184e-4 at ALL THREE meshes, because
%     stop.toleranceRule='meshScaled' makes tolerance proportional to sqrt(NE).

nO = numel(h.N);
P = struct();
P.outer  = (1:nO).';
P.omega1 = h.omega(1,:).';
P.omega2 = h.omega(2,:).';
P.gap12  = h.gap12(:);
P.volume = h.vol(:);
P.move   = h.move(:);
P.stage  = h.stage(:);
P.beta   = h.beta(:);
P.l2     = h.dxNorm2(:);
P.rms    = h.dxNorm2(:)/sqrt(NE);
P.maxAbs = h.dxOuter(:);
P.ratio  = h.dxOuter(:)./h.move(:);          % r_rho = max(u)
P.nInner = h.nInner(:);
P.innerConv = h.innerConv(:);
P.multN  = h.N(:);
P.degen  = h.degen(:);
P.descent     = [false; P.move(2:end) <  P.move(1:end-1)];
P.moveChanged = [true;  P.move(2:end) ~= P.move(1:end-1)];

% ---- discreteness -------------------------------------------------------
P.Mnd = zeros(nO,1); P.gray = zeros(nO,1); P.mid = zeros(nO,1);
for k = 1:nO
    r = RHO(:,k);
    P.Mnd(k)  = 100*mean(4*r.*(1-r));
    P.gray(k) = mean(r>0.1 & r<0.9);
    P.mid(k)  = mean(r>=0.4 & r<=0.6);
end

% ---- the FULL activity distribution ------------------------------------
qs = [0 0.50 0.75 0.90 0.95 0.975 0.99 1.00];
P.quantileLevels = qs;
P.uQuant = zeros(nO, numel(qs));
P.uMean  = zeros(nO,1);
P.uRMS   = zeros(nO,1);
P.Neff   = zeros(nO,1);

epsRMS = 8.83883476483184e-4;                 % = stop.tolerance/sqrt(NE), all meshes
P.tau  = [1e-4, epsRMS, 1e-3, 1e-2];
P.nActive = zeros(nO, numel(P.tau));

for k = 1:nO
    d = abs(DRHO(:,k));                       % raw |Delta rho|
    u = d / P.move(k);                        % normalised utilisation
    P.uQuant(k,:) = quantile(u, qs);
    P.uMean(k)    = mean(u);
    P.uRMS(k)     = sqrt(mean(u.^2));
    mx = max(d);
    if mx > 0; P.Neff(k) = (norm(d)/mx)^2; else; P.Neff(k) = NaN; end
    for j = 1:numel(P.tau)
        P.nActive(k,j) = sum(d > P.tau(j));   % STRICT >, as ms_run.m used
    end
end
P.NE = NE;
end
