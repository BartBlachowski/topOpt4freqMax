% test_passive_regions_rect.m
% Unit test for parsePassiveRegions: rect type.
%
% Grid:  nelx=10, nely=5, L=10, H=1
%   dx = 10/10 = 1.0,  dy = 1/5 = 0.2
%   Element centroids:
%     xc(elx) = (elx + 0.5) * 1.0   for elx = 0..9   -> 0.5, 1.5, ..., 9.5
%     yc(ely) = (ely + 0.5) * 0.2   for ely = 0..4   -> 0.1, 0.3, 0.5, 0.7, 0.9
%   Element index (1-based, column-major):  el = elx*nely + ely + 1
%
% rect: x0=[0, 0], size=[8, 0.1], is_solid=true
%   x-range: 0 <= xc <= 8   -> elx = 0..7  (xc = 0.5 .. 7.5)
%   y-range: 0 <= yc <= 0.1 -> ely = 0     (yc = 0.1)
%   Elements: ely=0 (row 0), elx=0..7  -> el = elx*5 + 0 + 1 = 1,6,11,16,21,26,31,36
%   Expected count: 8

fprintf('=== test_passive_regions_rect ===\n');

thisDir  = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(fullfile(repoRoot, 'tools'));

nelx = 10;  nely = 5;  L = 10.0;  H = 1.0;

cfg = struct();
cfg.domain.passive_regions.rect = struct( ...
    'x0',   [0.0, 0.0], ...
    'size', [8.0, 0.1], ...
    'is_solid', true);

[pasS, pasV] = parsePassiveRegions(cfg, nelx, nely, L, H);

fprintf('pasS count: %d\n', numel(pasS));
fprintf('pasS indices: %s\n', mat2str(pasS(:)'));
fprintf('pasV count: %d\n', numel(pasV));

%% Expected elements: ely=0, elx=0..7  -> el = elx*nely + 1 = [1 6 11 16 21 26 31 36]
expPasS = sort([1 6 11 16 21 26 31 36]');
expCount = 8;

assert(numel(pasV) == 0, ...
    'Test FAIL: expected 0 void elements, got %d', numel(pasV));
assert(numel(pasS) == expCount, ...
    'Test FAIL: expected %d solid elements, got %d', expCount, numel(pasS));
assert(isequal(sort(pasS(:)), expPasS), ...
    'Test FAIL: pasS indices mismatch. Got %s, expected %s', ...
    mat2str(pasS(:)'), mat2str(expPasS'));

fprintf('Test PASSED: %d solid elements, indices %s\n', numel(pasS), mat2str(pasS(:)'));

%% Test 2: void rect overlapping solid — solid wins
cfg2 = struct();
cfg2.domain.passive_regions.rect(1) = struct('x0', [0.0,0.0], 'size', [8.0,0.1], 'is_solid', true);
cfg2.domain.passive_regions.rect(2) = struct('x0', [0.0,0.0], 'size', [5.0,0.1], 'is_solid', false);

[pasS2, pasV2] = parsePassiveRegions(cfg2, nelx, nely, L, H);

% Void rect covers elx=0..4 (xc=0.5..4.5), ely=0. Solid covers elx=0..7, ely=0.
% Overlap: elx=0..4, ely=0 -> solid wins -> pasV2 should have NO overlap.
% pasS2: elx=0..7 ely=0 -> 8 elements.
% pasV2: elx=0..4 ely=0 overlap with pasS2 -> removed by solid-wins rule -> pasV2 empty.
assert(numel(pasS2) == 8, ...
    'Test 2 FAIL: expected 8 solid elements, got %d', numel(pasS2));
assert(numel(pasV2) == 0, ...
    'Test 2 FAIL: expected 0 void elements after solid-wins, got %d', numel(pasV2));

fprintf('Test 2 PASSED: solid-wins-over-void on overlap — %d solid, %d void\n', numel(pasS2), numel(pasV2));

%% Test 3: flanges (top + bottom)
cfg3 = struct();
cfg3.domain.passive_regions.flanges.bottom = struct('relative_height', 0.2, 'is_solid', true);
cfg3.domain.passive_regions.flanges.top    = struct('relative_height', 0.2, 'is_solid', true);

% nely=5, H=1, dy=0.2.  Bottom: yc <= 0.2 -> ely=0 (yc=0.1) -> 10 elements.
%                         Top:    yc >= 0.8 -> ely=4 (yc=0.9) -> 10 elements.
% Total: 20 solid, 0 void.
[pasS3, pasV3] = parsePassiveRegions(cfg3, nelx, nely, L, H);

assert(numel(pasV3) == 0, ...
    'Test 3 FAIL: expected 0 void elements, got %d', numel(pasV3));
assert(numel(pasS3) == 2*nelx, ...
    'Test 3 FAIL: expected %d flange elements, got %d', 2*nelx, numel(pasS3));

fprintf('Test 3 PASSED: flanges -> %d solid flange elements\n', numel(pasS3));

fprintf('\n=== All 3 tests PASSED ===\n');
