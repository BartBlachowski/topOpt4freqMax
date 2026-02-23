function [fixedDofs, dbg] = supportsToFixedDofs(supports, nelx, nely, L, H)
% SUPPORTSTOFIXEDDOFS  Convert new-style bc.supports entries to fixed DOF indices.
%
% Processes: vertical_line, horizontal_line, closest_point.
% Ignores:   hinge, clamp  (those are encoded in supportCode and applied by each
%            solver's own buildSupports / localBCAndLoad function).
%
% Node numbering  (column-major, matching all three solver conventions):
%   1-based node n at 0-based grid position (i=col, j=row):
%     n = i*(nely+1) + j + 1,   i = 0..nelx,  j = 0..nely
%     x = i*(L/nelx),           y = j*(H/nely)
%   DOFs (1-based):   ux = 2*n-1,   uy = 2*n

dx = L / nelx;
dy = H / nely;
N  = (nelx+1) * (nely+1);

% Build coordinate arrays for all nodes (vectorized, column-major order).
nodeIdx = (0:N-1)';
iArr    = floor(nodeIdx / (nely+1));   % 0-based column  (x-direction)
jArr    = mod(nodeIdx, nely+1);        % 0-based row     (y-direction)
nArr    = nodeIdx + 1;                 % 1-based node number
xArr    = iArr * dx;
yArr    = jArr * dy;

fixedDofs = [];
dbg = struct('entries', {{}});

for k = 1:numel(supports)
    s = supports(k);
    if ~isfield(s, 'type') || isempty(s.type), continue; end
    t = lower(strtrim(char(s.type)));

    switch t
        case {'hinge', 'clamp'}
            continue  % handled by each solver's own BC builder

        case 'vertical_line'
            x0  = double(s.x);
            tol = 1e-9;
            if isfield(s, 'tol') && ~isempty(s.tol), tol = double(s.tol); end
            mask = abs(xArr - x0) <= tol;

        case 'horizontal_line'
            y0  = double(s.y);
            tol = 1e-9;
            if isfield(s, 'tol') && ~isempty(s.tol), tol = double(s.tol); end
            mask = abs(yArr - y0) <= tol;

        case 'closest_point'
            loc   = double(s.location(:))';
            dist2 = (xArr - loc(1)).^2 + (yArr - loc(2)).^2;
            minD2 = min(dist2);
            % Tie-break: smallest node index (column-major order).
            selNode = min(nArr(dist2 == minD2));
            mask = (nArr == selNode);

        otherwise
            continue
    end

    selNodes  = nArr(mask);
    entryDofs = parseDofNames(s.dofs, selNodes);
    fixedDofs = [fixedDofs; entryDofs(:)];  %#ok<AGROW>

    dbg.entries{end+1} = struct('type', t, 'nodeCount', numel(selNodes), ...
        'nodes', selNodes(:)', 'dofs', entryDofs(:)');
    fprintf('  BC[%d] %s: %d nodes, %d dofs fixed\n', ...
        k, t, numel(selNodes), numel(entryDofs));
end

fixedDofs = unique(fixedDofs(:));
end

% -----------------------------------------------------------------------
function dofs = parseDofNames(dofsField, nodes)
% Convert "dofs" field (string / char / cell) to a column of 1-based DOF indices.
if ischar(dofsField)
    names = {dofsField};
elseif isstring(dofsField)
    names = cellstr(dofsField);
elseif iscell(dofsField)
    names = dofsField;
else
    names = {};
end
nodes = double(nodes(:));
dofs  = [];
for d = 1:numel(names)
    nm = lower(strtrim(char(names{d})));
    switch nm
        case 'ux', dofs = [dofs; 2*nodes - 1];
        case 'uy', dofs = [dofs; 2*nodes];
    end
end
end
