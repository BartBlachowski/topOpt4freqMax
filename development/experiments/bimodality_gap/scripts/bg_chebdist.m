function d = bg_chebdist(mask, cap)
%BG_CHEBDIST  Chebyshev (8-neighbour) distance in element units from every cell
%   to the nearest TRUE cell of MASK, computed by iterated 3x3 dilation (no
%   toolbox needed).  Distances beyond CAP are reported as CAP+1.
[ny, nx] = size(mask);
d = inf(ny, nx);
cur = mask;
d(cur) = 0;
for k = 1:cap
    p = false(ny+2, nx+2); p(2:end-1, 2:end-1) = cur;
    dil = false(ny, nx);
    for di = -1:1
        for dj = -1:1
            dil = dil | p((2:end-1)+di, (2:end-1)+dj);
        end
    end
    newly = dil & isinf(d);
    if ~any(newly(:)), break; end
    d(newly) = k;
    cur = dil;
end
d(isinf(d)) = cap + 1;
end
