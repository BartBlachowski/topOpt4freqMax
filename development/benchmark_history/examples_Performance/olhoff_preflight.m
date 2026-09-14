function [gates, admitted, methodIdx] = olhoff_preflight(resolutions, solverApproaches, opts)
%OLHOFF_PREFLIGHT  Decide, before any timing is measured, which Olhoff rows may
%   enter the performance tables.
%
%   [gates, admitted, methodIdx] = OLHOFF_PREFLIGHT(resolutions, solverApproaches)
%
%     gates      1 x nRes struct array of OLHOFF_EQUIVALENCE_GATE verdicts
%                (empty when no Olhoff column is present)
%     admitted   nRes x 1 logical; false means the row must not be run or
%                tabulated
%     methodIdx  column index of the Olhoff method, [] if absent
%
%   The Olhoff column is produced by a different implementation from the one
%   the column is named after: performance_comparison dispatches through
%   run_topopt_from_json -> OlhoffDu2007Repro -> run_repro2007 -> olhoffOpt, and
%   every hop can rename, default or override a setting.  A mapping defect in
%   that chain has already happened once and was silent -- both source JSONs
%   were valid, no error was raised, and the trajectory diverged at outer
%   iteration 101 (DIAGNOSTIC_REPRO2007_BENCHMARK.md).
%
%   So a timing row is not admitted because the run completed.  It is admitted
%   only when the dispatched path has been PROVED, for that mesh and this exact
%   profile, benchmark-path code, frozen implementation and MATLAB release, to
%   reproduce a direct call to the clean-room reproduction bit for bit.
%
%   This is a separate function rather than a block inside
%   PERFORMANCE_COMPARISON so that the thing standing between an unverified
%   number and a published table can itself be tested without running a
%   benchmark campaign.
%
%   opts is forwarded to OLHOFF_EQUIVALENCE_GATE.
%
%   See also OLHOFF_EQUIVALENCE_GATE, VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE,
%            PERFORMANCE_COMPARISON.

if nargin < 3; opts = struct(); end

nRes = size(resolutions, 1);
gates = struct('mesh', {}, 'status', {}, 'admissible', {}, 'row_class', {}, ...
    'reasons', {}, 'nelx', {}, 'nely', {});
admitted = true(nRes, 1);
methodIdx = find(strcmpi(solverApproaches, 'OlhoffDu2007Repro'));

if isempty(methodIdx)
    return
end

pm = 'r3';
if isfield(opts, 'profile_mode') && ~isempty(opts.profile_mode)
    pm = opts.profile_mode;
end
fprintf('\n--- Olhoff benchmark-path equivalence preflight (profile mode: %s) ---\n', pm);
for r = 1:nRes
    g = olhoff_equivalence_gate(resolutions(r,1), resolutions(r,2), opts);
    gates(end+1) = struct('mesh', g.mesh, 'status', g.status, ...
        'admissible', g.admissible, 'row_class', g.row_class, ...
        'reasons', {g.reasons}, 'nelx', g.nelx, 'nely', g.nely); %#ok<AGROW>
    admitted(r) = g.admissible;
    if g.admissible; tag = 'admitted'; else; tag = 'REFUSED'; end
    fprintf('  %-9s %-8s %-28s %s\n', g.mesh, g.status, g.row_class, tag);
    for q = 1:numel(g.reasons)
        fprintf('      %s\n', g.reasons{q});
    end
end

if ~all(admitted)
    fprintf(['\n  *** %d of %d Olhoff mesh(es) REFUSED.  Those rows are excluded from ' ...
             'the tables, scaling fits, ratios and speedups.\n'], sum(~admitted), nRes);
    fprintf(['      Remedy: run verify_repro2007_benchmark_equivalence and resolve any ' ...
             'FAIL before citing an Olhoff timing number.\n']);
end
fprintf('\n');
end
