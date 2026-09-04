function flt = prepFilter(nelx, nely, rmin)
%PREPFILTER  Sigmund (1997) mesh-independent filter weights, verbatim from
%   top88.m (Andreassen et al. 2011).  rmin is in ELEMENT units.
%
%   Element ordering is column-major, e = (i1-1)*nely + j1, matching model2D.
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
  for j1 = 1:nely
    e1 = (i1-1)*nely+j1;
    for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
      for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
        e2 = (i2-1)*nely+j2;
        k = k+1;
        iH(k) = e1;
        jH(k) = e2;
        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
      end
    end
  end
end
H  = sparse(iH,jH,sH);
Hs = sum(H,2);
flt = struct('H',H,'Hs',Hs,'rmin',rmin,'nelx',nelx,'nely',nely);
end
