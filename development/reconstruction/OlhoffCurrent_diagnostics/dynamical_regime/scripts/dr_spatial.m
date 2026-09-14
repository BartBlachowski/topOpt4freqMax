function S = dr_spatial(RHO, move, rho0, ks, nelx, nely, tag)
%DR_SPATIAL  Phase N -- where is the reversal, and what kind of element is it?
%   Element ordering el = elx*nely + ely (0-based) -> reshape(v, nely, nelx).

NE = nelx*nely;
X = [rho0*ones(NE,1), RHO]; D = diff(X,1,2);
S = struct('tag',tag,'ks',ks,'nelx',nelx,'nely',nely);
rows = [];
fprintf('\n=== spatial %s (%dx%d) ===\n', tag, nelx, nely);
fprintf('%6s %8s %8s %8s %8s %8s %8s %8s %8s\n', ...
    'k','revFrac','revVoid','revGray','revSolid','grayFrac','atBound','revAtBnd','clust');
for k = ks
    if k < 2 || k > size(D,2), continue; end
    a = D(:,k); b = D(:,k-1);
    rev = (a.*b) < 0;
    r   = RHO(:,k);
    void = r < 0.1; solid = r > 0.9; gray = ~void & ~solid;
    atB  = abs(a) > 0.9*move(k);
    % spatial clustering of reversers: fraction whose 4-neighbours also reverse
    M = reshape(rev, nely, nelx);
    nb = false(size(M));
    nb(2:end,:)   = nb(2:end,:)   | M(1:end-1,:);
    nb(1:end-1,:) = nb(1:end-1,:) | M(2:end,:);
    nb(:,2:end)   = nb(:,2:end)   | M(:,1:end-1);
    nb(:,1:end-1) = nb(:,1:end-1) | M(:,2:end);
    clust = mean(nb(M));                      % neighbourhood support of reversers
    row = [k, mean(rev), mean(rev(void)), mean(rev(gray)), mean(rev(solid)), ...
           mean(gray), mean(atB), mean(rev(atB)), clust];
    if isempty(rev(atB)), row(8) = NaN; end
    rows(end+1,:) = row; %#ok<AGROW>
    fprintf('%6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n', row);
end
S.cols = {'k','revFrac','revFrac_void','revFrac_gray','revFrac_solid','grayFrac', ...
          'boundFrac','revFrac_amongBound','reverserNeighbourSupport'};
S.table = rows;
end
