function [ok, E] = cs_socp_eligible(ctx, flags)
%CS_SOCP_ELIGIBLE  Preregistered eligibility contract E1-E8 (section 3),
%   evaluated BEFORE any solve.  Pure checks; no arithmetic reaches the design.
E = struct();
NE = numel(ctx.rho);
lam = ctx.lam(:);
E.N = flags.N; E.nlam = numel(lam);

E.E1 = flags.N == 2 && numel(lam) == 2 && (flags.n + flags.N) <= flags.Jcalc;

szF = size(ctx.F); if numel(szF) < 3, szF(3) = 1; end
E.E2 = isequal(szF, [NE 2 2]) && all(isfinite(ctx.F(:)));
if E.E2, E.E2 = isequal(ctx.F(:,1,2), ctx.F(:,2,1)); end

haveOff = isfield(ctx,'dOff') && ~isempty(ctx.dOff);
E.E3 = flags.useOff && haveOff && numel(lam) == 2;
if E.E3
    dOff = ctx.dOff(:);
    E.E3 = numel(dOff) == 2 && isequal(dOff, lam - lam(1)) && dOff(1) == 0 && dOff(2) >= 0;
end

E.E4 = E.E3 && all(abs((lam - ctx.dOff(:)) - lam(1)) <= 4*eps*lam(1));

E.E5 = numel(lam) == 2 && all(isfinite(lam)) && isfinite(ctx.lamJ) && lam(1) > 0 && ...
       lam(1) <= lam(2) && lam(2) <= ctx.lamJ && isequal(size(ctx.fJJ), [NE 1]) && ...
       all(isfinite(ctx.fJJ));

E.E6 = logical(flags.offDiag) && logical(ctx.offDiag);

noVolFun = ~isfield(ctx,'volFun') || isempty(ctx.volFun);
E.E7 = noVolFun && ~flags.useProj && flags.useSensFilter && ~flags.useDensityFilter && ...
       ~flags.innerLP && ~flags.innerDesignVar;

lo = max(ctx.rhomin - ctx.rho, -ctx.move);
hi = min(1 - ctx.rho, ctx.move);
E.E8 = ctx.move > 0 && all(lo <= 0) && all(hi >= 0) && all(ctx.rho >= ctx.rhomin) && all(ctx.rho <= 1);

ids = {'E1','E2','E3','E4','E5','E6','E7','E8'};
failed = ids(~cellfun(@(f) E.(f), ids));
ok = isempty(failed);
if ok, E.reason = ''; else, E.reason = ['eligibility failed: ' strjoin(failed, ',')]; end
E.ok = ok;
end
