function fixed = build_supports_exact(support_type, nodeNrs)
% BUILD_SUPPORTS_EXACT  Fixed DOFs for the Olhoff & Du (2014) beam examples.
%
%   fixed = build_supports_exact(support_type, nodeNrs)
%
%   Implements the three boundary conditions of Olhoff & Du (2014) Fig. 2:
%
%     'SS'  (Fig. 2a)  Simply supported ends.  Pin (ux and uy) at the
%                      MID-HEIGHT node of each vertical edge.
%     'CS'  (Fig. 2b)  Left edge fully clamped; right edge pinned at mid-height.
%     'CC'  (Fig. 2c)  Both vertical edges fully clamped.
%
%   Support location for SS/CS
%   --------------------------
%   Verified against Fig. 2(a) rendered at 300 dpi: the support triangle apex
%   sits ON the vertical end edge at approximately 0.45-0.50 b, i.e. at
%   mid-height, NOT at the bottom corner.  A bottom-corner pin engages arch
%   action and gives omega_1 ~ 100 rad/s for the uniform initial design instead
%   of the published 68.7.
%
%   Because the pin must land exactly on y = H/2, NELY MUST BE EVEN.  With odd
%   nely there is no node at mid-height; the historical implementation used
%   mid_idx = round(nely/2)+1, and MATLAB rounds 2.5 away from zero, so at
%   nely = 5 the pin landed on node row 4 (y = 0.600 H).  The SS and CS
%   problems are then not mirror-symmetric -- a different structural problem,
%   not a coarse approximation of the paper's.
%
%   nodeNrs: (nely+1) x (nelx+1) matrix of 1-based node numbers,
%       nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1)
%   Row 1 is y = 0 (bottom), row nely+1 is y = H (top).
%
%   DOF convention (1-based):  ux(n) = 2n-1,  uy(n) = 2n.
%
%   Reference: Olhoff & Du (2014), CISM 2014, Fig. 2.

nely = size(nodeNrs,1) - 1;

left_nodes  = nodeNrs(:, 1);
right_nodes = nodeNrs(:, end);

u = @(n) 2*n - 1;   % ux DOF
v = @(n) 2*n;       % uy DOF

stype = upper(strtrim(char(support_type)));

if any(strcmp(stype, {'SS','CS'}))
    if mod(nely, 2) ~= 0
        error('build_supports_exact:OddNely', ...
            ['support_type ''%s'' needs a node at mid-height, so nely must be ' ...
             'EVEN (got nely = %d).  With odd nely the pin lands off-centre ' ...
             '(nely = 5 puts it at y = 0.600 H) and the problem is no longer ' ...
             'mirror-symmetric.  See PLAN_Olhoff2014_exact.md finding F3.'], ...
            stype, nely);
    end
    mid_idx   = nely/2 + 1;          % exact mid-height row, 1-based
    left_mid  = nodeNrs(mid_idx, 1);
    right_mid = nodeNrs(mid_idx, end);
end

switch stype

    case 'SS'
        % Fig. 2a: pin at mid-height of both vertical edges.
        fixed = [u(left_mid);  v(left_mid); ...
                 u(right_mid); v(right_mid)];

    case 'CS'
        % Fig. 2b: left edge clamped, right edge pinned at mid-height.
        fixed = [u(left_nodes(:)); v(left_nodes(:)); ...
                 u(right_mid); v(right_mid)];

    case 'CC'
        % Fig. 2c: both vertical edges fully clamped.
        fixed = [u(left_nodes(:)); v(left_nodes(:)); ...
                 u(right_nodes(:)); v(right_nodes(:))];

    otherwise
        error('build_supports_exact:UnknownType', ...
            'build_supports_exact: unknown support_type ''%s''. Use SS, CS or CC.', ...
            support_type);
end

fixed = unique(fixed(:));
end
