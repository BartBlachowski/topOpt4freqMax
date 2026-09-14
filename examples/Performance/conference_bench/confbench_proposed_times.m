function T = confbench_proposed_times(tInit, tEig, tSimp, callWall, selfWall)
%CONFBENCH_PROPOSED_TIMES  The Proposed method's time struct (timing schema 2).
%
%   T = CONFBENCH_PROPOSED_TIMES(tInit, tEig, tSimp, callWall, selfWall)
%
%     tInit     solver initialization (tel.timing.initialization_time): mesh,
%               element matrices, connectivity, filter, boundary conditions AND
%               the reference eigenanalysis
%     tEig      the reference eigenanalysis alone: K0/M0 assembly, the
%               eigensolve and extraction of the frozen mode(s); a SUB-INTERVAL
%               of tInit  (tel.timing.stage1_reference_eigen_time)
%     tSimp     the SIMP loop (tel.timing.optimization_loop_time)
%     callWall  caller-side tic/toc around the whole solve
%     selfWall  the solver's own self-reported wall time
%
%   Time 1 is the reference eigenanalysis and NOTHING else.  The preparation
%   (tInit - tEig) is one-off setup, not a computational stage of the method,
%   and it is accounted in overhead_time_s -- the table's "Other" column --
%   exactly where the Yuksel and Du-Olhoff records put their own setup, so that
%   Other means the same thing on every row and Time 1 + Time 2 + Other = Total.
%
%   Timing schema 1 folded the preparation into Time 1.  That made a 0.03 s
%   eigensolve read as 0.43 s at 160x20, next to a 0.035 s Du-Olhoff
%   per-outer-iteration cost that excludes setup.  Schema 2 (2026-09-14) fixes
%   the asymmetry here, in the ONE place the Proposed accounting is defined:
%   CONFBENCH_RUN_CASE uses it for a fresh solve, and
%   COMPOSE_NINE_MESH_COMPARISON uses it to bring recorded schema-1 rows to
%   schema 2 from their recorded sub-interval timers, without re-timing.
%
%   See also CONFBENCH_RUN_CASE, CONFBENCH_TIMING_SCHEMA, CONFBENCH_ACCOUNTING.

assert(isfinite(tEig) && tEig >= 0 && tEig <= tInit + 1e-9, ...
    'confbench_proposed_times:Nesting', ...
    ['The reference eigenanalysis (%.6f s) is not a sub-interval of the ' ...
     'solver initialization (%.6f s).'], tEig, tInit);
assert(isfinite(tSimp) && tSimp >= 0 && isfinite(callWall) && callWall >= tInit + tSimp - 1e-9, ...
    'confbench_proposed_times:Total', ...
    'Initialization (%.6f s) + SIMP (%.6f s) exceed the caller-side total (%.6f s).', ...
    tInit, tSimp, callWall);

T = struct( ...
    'time1_name', 'stage1_reference_eigenanalysis_s', ...
    'time1', tEig, ...
    'time2_name', 'stage2_simp_time_s', ...
    'time2', tSimp, ...
    'stage1_time_s', tInit, ...                 % initialization INCLUDING the eigenanalysis (schema-1 Time 1)
    'stage1_reference_eigen_time_s', tEig, ...
    'stage1_preparation_time_s', tInit - tEig, ...   % in overhead_time_s, never in Time 1
    'stage2_time_s', tSimp, ...
    'overhead_time_s', callWall - tEig - tSimp, ...
    'total_wall_time_s', callWall, ...
    'solver_self_report_wall_s', selfWall, ...
    'independent_crosscheck_residual_s', callWall - selfWall);
end
