% TASK1_SUPPORT_GEOMETRY_AUDIT
%
% READ-ONLY audit of the Du & Olhoff (2007) benchmark discretization as it is
% currently defined in analysis/OlhoffApproachExact/Matlab.
%
% This script calls the PRODUCTION function build_supports_exact() verbatim and
% reports, for each candidate mesh, the exact physical coordinates of every
% constrained node.  It writes NOTHING back into the solver and changes no
% state.  Its only purpose is to answer TASK 1:
%
%   - current mesh
%   - beam aspect ratio
%   - support coordinates
%   - exact node selection used for supports
%   - is the support exactly at H/2, or shifted by odd nely?
%
% Output: results/task1_support_geometry.csv  +  console table.

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));

out_dir = fullfile(this_dir, 'results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% Benchmark geometry, from topopt_freq_exact.m set_defaults() -- unchanged.
L = 8.0;
H = 1.0;

meshes = [40 5; 80 10; 160 20; 240 30];
bcs    = {'SS', 'CS', 'CC'};

fid = fopen(fullfile(out_dir, 'task1_support_geometry.csv'), 'w');
fprintf(fid, ['nelx,nely,L,H,aspect_ratio,dx,dy,elem_aspect,nEl,nNode,nDof,', ...
              'bc,nFixedDof,mid_idx,mid_y_over_H,mid_offset_over_H,exact_midheight\n']);

fprintf('\n');
fprintf('==================================================================\n');
fprintf(' TASK 1 -- Benchmark discretization + support geometry audit\n');
fprintf(' Geometry from topopt_freq_exact.m defaults: L=%.4f  H=%.4f\n', L, H);
fprintf('==================================================================\n');

for mi = 1:size(meshes, 1)
    nelx = meshes(mi, 1);
    nely = meshes(mi, 2);

    dx = L / nelx;
    dy = H / nely;
    nEl   = nelx * nely;
    nNode = (nelx + 1) * (nely + 1);
    nDof  = 2 * nNode;

    % EXACT reproduction of topopt_freq_exact.m line 166.
    nodeNrs = reshape(1:nNode, nely + 1, nelx + 1);

    % EXACT reproduction of build_supports_exact.m line 34.
    mid_idx = round(nely / 2) + 1;
    mid_y   = (mid_idx - 1) * dy;
    offset  = (mid_y - H / 2) / H;
    exact   = abs(offset) < 1e-12;

    fprintf('\n------------------------------------------------------------------\n');
    fprintf(' MESH %dx%d   L/H = %.4f   dx=%.6f dy=%.6f  elem aspect dx/dy=%.6f\n', ...
        nelx, nely, L / H, dx, dy, dx / dy);
    fprintf(' nEl=%d  nNode=%d  nDof=%d\n', nEl, nNode, nDof);
    fprintf(' Node-row count = nely+1 = %d   (rows are y = 0, dy, ..., H)\n', nely + 1);
    fprintf(' build_supports_exact mid_idx = round(nely/2)+1 = %d\n', mid_idx);
    fprintf(' -> mid-height node at y = %.8f  (H/2 = %.8f)\n', mid_y, H / 2);
    if exact
        fprintf(' -> EXACT mid-height. offset = 0.\n');
    else
        fprintf(' -> SHIFTED. offset = %+.6f * H  (=%+.6f in absolute units)\n', ...
            offset, offset * H);
        fprintf('    Cause: nely=%d is ODD, so no node lies on y=H/2;\n', nely);
        fprintf('    MATLAB round() rounds 0.5 away from zero -> row %d not %d.\n', ...
            mid_idx, floor(nely / 2) + 1);
    end

    for bi = 1:numel(bcs)
        bc = bcs{bi};
        fixed = build_supports_exact(bc, nodeNrs);

        % Decode: 1-based DOF -> node -> (ix, iy) -> (x, y)
        nodes = ceil(fixed(:) / 2);
        comp  = 2 - mod(fixed(:), 2);          % 1 = ux, 2 = uy
        iy    = mod(nodes - 1, nely + 1) + 1;  % column-major node numbering
        ix    = floor((nodes - 1) / (nely + 1)) + 1;
        xs    = (ix - 1) * dx;
        ys    = (iy - 1) * dy;

        uniq_nodes = unique(nodes);
        fprintf('   [%s] nFixedDof=%3d over %d node(s):\n', bc, numel(fixed), numel(uniq_nodes));
        for k = 1:numel(uniq_nodes)
            n = uniq_nodes(k);
            sel = nodes == n;
            cs = comp(sel);
            tag = '';
            if any(cs == 1), tag = [tag 'ux']; end
            if any(cs == 2), if ~isempty(tag), tag = [tag '+']; end, tag = [tag 'uy']; end
            if numel(uniq_nodes) <= 8 || k <= 2 || k > numel(uniq_nodes) - 2
                fprintf('        node %6d  (x=%.6f, y=%.6f)  y/H=%.6f  [%s]\n', ...
                    n, xs(find(sel, 1)), ys(find(sel, 1)), ys(find(sel, 1)) / H, tag);
            elseif k == 3
                fprintf('        ... (%d further nodes on the clamped edge) ...\n', ...
                    numel(uniq_nodes) - 4);
            end
        end

        fprintf(fid, '%d,%d,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%d,%d,%d,%s,%d,%d,%.10f,%.10f,%d\n', ...
            nelx, nely, L, H, L / H, dx, dy, dx / dy, nEl, nNode, nDof, ...
            bc, numel(fixed), mid_idx, mid_y / H, offset, exact);
    end
end

fclose(fid);
fprintf('\n==================================================================\n');
fprintf(' Wrote %s\n', fullfile(out_dir, 'task1_support_geometry.csv'));
fprintf('==================================================================\n\n');
