function M = stopstudy_metrics(X, x0)
%STOPSTUDY_METRICS per-iteration design-change and discreteness metrics.
%  X  : n_e x n_iter physical densities AFTER each update
%  x0 : n_e x 1 design before the first update
n = size(X,2);
M = struct();
z = nan(n,1);
M.dmax=z; M.d2=z; M.drel=z; M.drms=z; M.dmean=z; M.Mnd=z; M.gray=z; M.flip=z; M.vol=z; M.dMnd=z;
prev = x0;
for k = 1:n
    x = X(:,k); d = x - prev;
    M.dmax(k) = max(abs(d));
    M.d2(k)   = norm(d);
    M.drel(k) = norm(d)/max(norm(prev), realmin);
    M.drms(k) = sqrt(mean(d.^2));
    M.dmean(k)= mean(abs(d));
    M.Mnd(k)  = mean(4*x.*(1-x));
    M.gray(k) = mean(x > 0.1 & x < 0.9);
    M.flip(k) = mean((x > 0.5) ~= (prev > 0.5));
    M.vol(k)  = mean(x);
    M.dMnd(k) = M.Mnd(k) - mean(4*prev.*(1-prev));
    prev = x;
end
end
