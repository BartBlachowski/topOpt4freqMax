function fixed = build_supports_exact(support_type, nodeNrs)
% BUILD_SUPPORTS_EXACT  Fixed DOFs for Du & Olhoff (2007) beam examples.
%
%   fixed = build_supports_exact(support_type, nodeNrs)
%
%   Implements the three boundary conditions from paper Figure 2:
%
%     'SS'  Simply-supported at both ends.
%           Pin (ux+uy fixed) at the mid-height node of each vertical edge.
%
%     'CS'  Clamped-Simply: left edge fully clamped, right edge has a pin
%           at mid-height (per Fig. 2b).
%
%     'CC'  Clamped-Clamped: both vertical edges fully clamped.
%
%   nodeNrs: (nely+1) x (nelx+1) matrix of 1-based node numbers arranged
%   in column-major order, i.e.
%       nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1)
%   Row 1 is the bottom edge (y = 0), row nely+1 is the top edge (y = H).
%
%   DOF convention (1-based, consistent with the rest of the codebase):
%       ux(n) = 2*n - 1
%       uy(n) = 2*n
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110, Fig. 2.

nely = size(nodeNrs,1) - 1;

left_nodes  = nodeNrs(:, 1);
right_nodes = nodeNrs(:, end);

% Mid-height node: row index at y = H/2.
% round(nely/2)+1 gives the central row in a 1-based (nely+1)-row node array.
mid_idx   = round(nely/2) + 1;
left_mid  = nodeNrs(mid_idx, 1);
right_mid = nodeNrs(mid_idx, end);

u = @(n) 2*n - 1;   % ux DOF, 1-based
v = @(n) 2*n;       % uy DOF, 1-based

switch upper(strtrim(char(support_type)))

    case 'SS'
        % Simply-supported: pin at mid-height of both edges.
        fixed = [u(left_mid);  v(left_mid); ...
                 u(right_mid); v(right_mid)];

    case 'CS'
        % Left edge clamped, right edge pinned at mid-height.
        fixed = [u(left_nodes(:)); v(left_nodes(:)); ...
                 u(right_mid); v(right_mid)];

    case 'CC'
        % Both edges fully clamped.
        fixed = [u(left_nodes(:)); v(left_nodes(:)); ...
                 u(right_nodes(:)); v(right_nodes(:))];

    otherwise
        error('build_supports_exact: unknown support_type ''%s''. Use SS, CS, or CC.', support_type);
end

fixed = unique(fixed(:));
end
