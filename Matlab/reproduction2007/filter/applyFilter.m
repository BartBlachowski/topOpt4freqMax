function df = applyFilter(flt, rho, df)
%APPLYFILTER  top88 sensitivity filter (ft = 1), applied column-wise.
%
%   df may hold several sensitivity vectors as columns (e.g. the f_sk of
%   Du & Olhoff eq. 19); each is filtered independently.
%
%   The max(1e-3, x) guard is top88's published normalisation and is kept
%   deliberately -- see CLAUDE.md sec.6.
rho = rho(:);
den = flt.Hs .* max(1e-3, rho);
for c = 1:size(df,2)
    df(:,c) = (flt.H * (rho .* df(:,c))) ./ den;
end
end
