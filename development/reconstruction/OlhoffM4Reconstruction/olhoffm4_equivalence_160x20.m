function report = olhoffm4_equivalence_160x20(outFile)
%OLHOFFM4_EQUIVALENCE_160X20  Prove the import reproduces the audited source.
%
%   Three runs of the SAME frozen configuration at 160x20 -- the project's
%   documented mesh-resolution floor, so this is a scientific-scale proof and
%   not a mechanics check:
%
%     A  the AUDITED SOURCE tree /Users/piotrek/Programming/Matlab/Olhoff,
%        configuration read straight out of
%        audit_termination_mesh_admission/runs/TMA_FROZEN_CFGS.mat
%     B  the IMPORT under analysis/OlhoffM4Reconstruction, same configuration
%     C  the IMPORT under the benchmark configuration, which differs from the
%        audited one in exactly one scientific field: diag = false
%
%   A vs B proves the import -- including the timing-instrumentation patch to
%   olhoffOpt.m -- is behaviourally inert.  B vs C proves that switching the
%   per-iteration diagnostic recorder off, which the benchmark does so it is
%   not timed, leaves the trajectory bit-for-bit unchanged.
%
%   Equality is BITWISE on the raw IEEE-754 bytes of the final density field.
%   "Agrees to four figures" is exactly what this must not accept.

if nargin < 1 || isempty(outFile)
    outFile = fullfile(olhoffm4_root(), 'evidence', 'import_equivalence_160x20.json');
end
root = olhoffm4_root();
src  = '/Users/piotrek/Programming/Matlab/Olhoff';
entryPath = path();
restore = onCleanup(@() path(entryPath)); %#ok<NASGU>

L = load(fullfile(src, 'audit_termination_mesh_admission', 'runs', 'TMA_FROZEN_CFGS.mat'));
cfgAudited = L.CFG(1).cfg;
assert(cfgAudited.nelx == 160 && cfgAudited.nely == 20, ...
    'olhoffm4_equivalence:WrongMesh', 'Expected the 160x20 frozen configuration.');
cfgBench = olhoffm4_config(160, 20);

report = struct();
report.generated = char(string(datetime('now','TimeZone','local','Format','yyyy-MM-dd''T''HH:mm:ssXXX')));
report.mesh = [160 20];
report.source_repository = src;
report.note = ['Bitwise comparison of the final density field. Run A is the ' ...
    'audited source tree, run B the import under the same configuration, ' ...
    'run C the import under the benchmark configuration (diag = false).'];

% ---- A: the audited source ---------------------------------------------
fprintf('[A] audited source tree, diag=%d ...\n', cfgAudited.diag);
restoredefaultpath();
addpath(fullfile(src,'algo'), fullfile(src,'fem'), fullfile(src,'filter'));
maxNumCompThreads(1);
assert(strncmp(which('olhoffOpt'), src, numel(src)), 'run A did not resolve to the source');
tA = tic; resA = olhoffOpt(cfgAudited); wA = toc(tA);
report.A = runSummary(resA, wA, which('olhoffOpt'), cfgAudited);

% ---- B: the import, identical configuration -----------------------------
fprintf('[B] imported tree, identical configuration ...\n');
restoredefaultpath();
addpath(root);
maxNumCompThreads(1);
gB = olhoffm4_paths();   %#ok<NASGU>  held alive until the explicit clear below
assert(strncmp(which('olhoffOpt'), root, numel(root)), 'run B did not resolve to the import');
tB = tic; resB = olhoffOpt(cfgAudited); wB = toc(tB);
report.B = runSummary(resB, wB, which('olhoffOpt'), cfgAudited);
clear gB

% ---- C: the import, benchmark configuration (diag = false) --------------
fprintf('[C] imported tree, benchmark configuration (diag = false) ...\n');
restoredefaultpath();
addpath(root);
maxNumCompThreads(1);
gC = olhoffm4_paths();   %#ok<NASGU>  held alive until the explicit clear below
tC = tic; resC = olhoffOpt(cfgBench); wC = toc(tC);
report.C = runSummary(resC, wC, which('olhoffOpt'), cfgBench);
clear gC

% ---- the comparisons ----------------------------------------------------
report.A_vs_B = compareRuns(resA, resB);
report.B_vs_C = compareRuns(resB, resC);
report.A_vs_C = compareRuns(resA, resC);
report.pass = report.A_vs_B.bitwise_identical && report.B_vs_C.bitwise_identical;

% ---- the configuration differences between audited and benchmark --------
fn = union(fieldnames(cfgAudited), fieldnames(cfgBench));
diffs = {};
for i = 1:numel(fn)
    f = fn{i};
    if ~isfield(cfgAudited,f) || ~isfield(cfgBench,f) || ~isequaln(cfgAudited.(f), cfgBench.(f))
        diffs{end+1} = f; %#ok<AGROW>
    end
end
report.audited_vs_benchmark_config_differences = diffs;
report.expected_differences = {'diag','name','mmasubPath'};
report.only_expected_differences = isempty(setdiff(diffs, report.expected_differences));
report.pass = report.pass && report.only_expected_differences;

restoredefaultpath(); addpath(root);
od = fileparts(outFile);
if exist(od,'dir') ~= 7; mkdir(od); end
fid = fopen(outFile,'w'); c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(report, 'PrettyPrint', true));

fprintf('\n  A  rho %s  omega1 = %.10f  outer = %d  cumInner = %d  wall = %.1f s\n', report.A.rho_sha256(1:16), report.A.omega1, report.A.nOuter, report.A.cum_inner, wA);
fprintf('  B  rho %s  omega1 = %.10f  outer = %d  cumInner = %d  wall = %.1f s\n', report.B.rho_sha256(1:16), report.B.omega1, report.B.nOuter, report.B.cum_inner, wB);
fprintf('  C  rho %s  omega1 = %.10f  outer = %d  cumInner = %d  wall = %.1f s\n', report.C.rho_sha256(1:16), report.C.omega1, report.C.nOuter, report.C.cum_inner, wC);
fprintf('  A vs B bitwise identical : %d\n', report.A_vs_B.bitwise_identical);
fprintf('  B vs C bitwise identical : %d\n', report.B_vs_C.bitwise_identical);
fprintf('  audited -> benchmark cfg differences: %s (only expected: %d)\n', strjoin(diffs, ', '), report.only_expected_differences);
fprintf('  EQUIVALENCE PASS: %d\n', report.pass);
fprintf('  written to %s\n', outFile);
end

% =========================================================================
function s = runSummary(res, wall, implFile, cfg)
s = struct();
s.implementation = implFile;
s.config_name = cfg.name;
s.diag = cfg.diag;
s.nOuter = res.nOuter;
s.omega1 = res.omega(1);
s.omega2 = res.omega(2);
s.omega3 = res.omega(3);
s.volume = mean(res.rho);
s.cum_inner = res.hist.cumInner(end);
s.final_move = res.hist.move(end);
s.wall_s = wall;
s.rho_sha256 = sha256Raw(res.rho(:));
s.omega_sha256 = sha256Raw(res.omega(:));
s.converged = any(contains(res.log, 'converged at outer'));
end

function c = compareRuns(r1, r2)
c = struct();
c.rho_bitwise = isequal(typecast(r1.rho(:),'uint8'), typecast(r2.rho(:),'uint8'));
c.omega_bitwise = isequal(typecast(r1.omega(:),'uint8'), typecast(r2.omega(:),'uint8'));
c.nOuter_equal = r1.nOuter == r2.nOuter;
c.max_abs_rho_difference = max(abs(r1.rho(:) - r2.rho(:)));
flds = {'N','beta','nInner','dxOuter','dxNorm2','move','stage','cumInner','vol'};
same = true;
for i = 1:numel(flds)
    same = same && isequaln(r1.hist.(flds{i})(:), r2.hist.(flds{i})(:));
end
c.history_identical = same;
c.bitwise_identical = c.rho_bitwise && c.omega_bitwise && c.nOuter_equal && same;
end

function h = sha256Raw(v)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
