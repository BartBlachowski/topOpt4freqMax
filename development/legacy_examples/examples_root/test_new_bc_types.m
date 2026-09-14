% test_new_bc_types.m
% Unit test for the supportsToFixedDofs helper introduced in
% tools/run_topopt_from_json.m.
%
% Grid:  nelx=4, nely=2, L=4, H=2
%   Nodes per edge:  left/right: nely+1 = 3
%                    bottom/top: nelx+1 = 5
%   Node numbering (column-major, i=col 0..nelx, j=row 0..nely):
%     n = i*(nely+1) + j + 1
%   Grid layout (node numbers):
%       j=2:  3  6  9  12  15
%       j=1:  2  5  8  11  14
%       j=0:  1  4  7  10  13
%            i=0 1  2   3   4

fprintf('=== test_new_bc_types ===\n');

nelx = 4;  nely = 2;  L = 4.0;  H = 2.0;

% Make sure the helper is visible.
thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(fullfile(repoRoot, 'tools'));

%% ---- Test 1: vertical_line x=0, dofs=[ux,uy] -------------------------
% Expects: all nodes on left edge (i=0, j=0..nely) = nely+1 = 3 nodes
%          node IDs: 1, 2, 3
%          DOFs: 1,2,3,4,5,6  (ux and uy of each node)
s1 = struct('type', 'vertical_line', 'x', 0.0, 'tol', 1e-9, ...
            'dofs', {{"ux","uy"}});
[fd1, dbg1] = supportsToFixedDofs(s1, nelx, nely, L, H);

assert(dbg1.entries{1}.nodeCount == nely+1, ...
    'Test 1 FAIL: expected %d nodes, got %d', nely+1, dbg1.entries{1}.nodeCount);
assert(numel(fd1) == 2*(nely+1), ...
    'Test 1 FAIL: expected %d dofs, got %d', 2*(nely+1), numel(fd1));
assert(isequal(sort(dbg1.entries{1}.nodes(:))', 1:nely+1), ...
    'Test 1 FAIL: wrong node IDs');

fprintf('Test 1 PASSED: vertical_line x=0  -> %d nodes (IDs: %s), %d dofs\n', ...
    dbg1.entries{1}.nodeCount, mat2str(sort(dbg1.entries{1}.nodes)), numel(fd1));

%% ---- Test 2: horizontal_line y=0, dofs=[uy] --------------------------
% Expects: all nodes on bottom edge (j=0, i=0..nelx) = nelx+1 = 5 nodes
%          node IDs: 1, 4, 7, 10, 13  (n = i*(nely+1)+0+1 = i*3+1)
%          DOFs: uy only -> 5 dofs
s2 = struct('type', 'horizontal_line', 'y', 0.0, 'tol', 1e-9, ...
            'dofs', {{"uy"}});
[fd2, dbg2] = supportsToFixedDofs(s2, nelx, nely, L, H);

expNodes2 = (0:nelx)*(nely+1) + 1;   % = [1 4 7 10 13]
assert(dbg2.entries{1}.nodeCount == nelx+1, ...
    'Test 2 FAIL: expected %d nodes, got %d', nelx+1, dbg2.entries{1}.nodeCount);
assert(numel(fd2) == nelx+1, ...
    'Test 2 FAIL: expected %d dofs, got %d', nelx+1, numel(fd2));
assert(isequal(sort(dbg2.entries{1}.nodes(:))', sort(expNodes2)), ...
    'Test 2 FAIL: wrong node IDs');

fprintf('Test 2 PASSED: horizontal_line y=0 -> %d nodes (IDs: %s), %d dofs (uy only)\n', ...
    dbg2.entries{1}.nodeCount, mat2str(sort(dbg2.entries{1}.nodes)), numel(fd2));

%% ---- Test 3: closest_point [0,0], dofs=[ux,uy] -----------------------
% Expects: single node at (x=0, y=0) = node 1
%          DOFs: ux=1, uy=2
s3 = struct('type', 'closest_point', 'location', [0.0, 0.0], ...
            'dofs', {{"ux","uy"}});
[fd3, dbg3] = supportsToFixedDofs(s3, nelx, nely, L, H);

assert(dbg3.entries{1}.nodeCount == 1, ...
    'Test 3 FAIL: expected 1 node, got %d', dbg3.entries{1}.nodeCount);
assert(dbg3.entries{1}.nodes(1) == 1, ...
    'Test 3 FAIL: expected node 1, got %d', dbg3.entries{1}.nodes(1));
assert(isequal(sort(fd3(:))', [1 2]), ...
    'Test 3 FAIL: wrong dofs, got %s', mat2str(sort(fd3(:))'));

fprintf('Test 3 PASSED: closest_point [0,0] -> node 1, dofs %s\n', ...
    mat2str(sort(fd3(:))'));

%% ---- Test 4: hinge/clamp entries are silently ignored -----------------
% Both structs must share the same field set for MATLAB array concatenation
% (jsondecode does this automatically for JSON arrays).
s4a = struct('type', 'hinge',         'location', 'left_mid_height', 'x', [], 'tol', [], 'dofs', {{"ux","uy"}});
s4b = struct('type', 'vertical_line', 'location', [],                'x', 0.0, 'tol', 1e-9, 'dofs', {{"ux"}});
supports4 = [s4a, s4b];
[fd4, dbg4] = supportsToFixedDofs(supports4, nelx, nely, L, H);

assert(numel(dbg4.entries) == 1, ...
    'Test 4 FAIL: expected 1 processed entry (hinge ignored), got %d', numel(dbg4.entries));
assert(numel(fd4) == nely+1, ...
    'Test 4 FAIL: expected %d dofs (ux of left edge), got %d', nely+1, numel(fd4));

fprintf('Test 4 PASSED: hinge entry ignored, vertical_line ux-only -> %d dofs\n', numel(fd4));

%% ---- Test 5: duplicate dofs are deduplicated -------------------------
s5a = struct('type', 'vertical_line',   'x', 0.0, 'y', [],  'tol', 1e-9, 'dofs', {{"ux","uy"}});
s5b = struct('type', 'horizontal_line', 'x', [],  'y', 0.0, 'tol', 1e-9, 'dofs', {{"uy"}});
supports5 = [s5a, s5b];
[fd5, ~] = supportsToFixedDofs(supports5, nelx, nely, L, H);

% Left-edge nodes: 1,2,3 -> 6 dofs
% Bottom-edge nodes: 1,4,7,10,13 -> uy dofs = 2,8,14,20,26
% Node 1 uy=2 counted in both -> after unique: 6 + 4 = 10 dofs
expCount5 = 2*(nely+1) + nelx;   % = 6 + 4
assert(numel(fd5) == expCount5, ...
    'Test 5 FAIL: expected %d unique dofs, got %d', expCount5, numel(fd5));

fprintf('Test 5 PASSED: deduplication -> %d unique dofs\n', numel(fd5));

fprintf('\n=== All 5 tests PASSED ===\n');
