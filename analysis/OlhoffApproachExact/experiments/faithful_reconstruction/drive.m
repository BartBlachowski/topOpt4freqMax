function drive(series, nelx, nely)
% DRIVE  Batch driver for the faithful-reconstruction campaign.
%
%   drive(series, nelx, nely)
%
%   series
%     'alphaA'  V0,V1  @ inner_max_iter = 30   (recorded budget, paper-literal)
%     'alphaB'  V2,V3  @ inner_max_iter = 30   (recorded budget, fail-closed)
%     'alphaC'  V4,V5  @ inner_max_iter = 30   (recorded budget, Regime-B + FC)
%     'betaA'   V0,V1  @ inner_max_iter = 2000 (paper-faithful inner budget)
%     'betaB'   V2,V3  @ inner_max_iter = 2000
%     'betaC'   V4,V5  @ inner_max_iter = 2000
%     'sched'   V1a,V1b @ 2000  (continuation-schedule sensitivity, Phase 3)
%
%   The two inner budgets are BOTH reported.  30 is the value recorded in the
%   on-disk regimes; 2000 is the smallest round budget that lets the inner
%   subproblem actually reach its declared tolerance, which is what Du &
%   Olhoff's Fig. 1 inner loop requires ("increments converged?").

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
maxNumCompThreads(3);
if nargin < 2, nelx = 160; nely = 20; end

% The paper-literal beta variants (V0..V3 at the converged inner budget) are
% given 15 outer iterations rather than 300: their terminal classification is
% MECHANISM_COLLAPSE, established at outer iteration 1 and never reversed, and
% every p on the continuation path is additionally probed one step at a time by
% phase3_continuation_probe.m.  Their continuation stages are shortened to 3 so
% that all five p values are still exercised inside the budget.  The i30 series
% runs the same four variants for the full 300 outer iterations and confirms the
% collapse is never reversed; the short i2000 budget exists only to show that a
% GENUINELY CONVERGED inner solve collapses too.  eigs on the resulting
% near-singular K is what makes these runs expensive, not the optimizer.
switch series
    case 'alphaA', vars = {'V0','V1'};   ib = 30;   sfx = '_i30';   om = 300; sl = 25;
    case 'alphaB', vars = {'V2','V3'};   ib = 30;   sfx = '_i30';   om = 300; sl = 25;
    case 'alphaC', vars = {'V4','V5'};   ib = 30;   sfx = '_i30';   om = 300; sl = 25;
    case 'betaA',  vars = {'V0','V1'};   ib = 2000; sfx = '_i2000'; om = 15;  sl = 3;
    case 'betaB',  vars = {'V2','V3'};   ib = 2000; sfx = '_i2000'; om = 15;  sl = 3;
    case 'betaC',  vars = {'V4','V5'};   ib = 2000; sfx = '_i2000'; om = 300; sl = 25;
    case 'sched',  vars = {'V5a','V5b'}; ib = 2000; sfx = '_i2000'; om = 300; sl = 25;
    otherwise, error('drive: unknown series %s', series);
end

for k = 1:numel(vars)
    o = struct('outer_max_iter', om, 'inner_max_iter', ib, ...
               'cont_stage_len', sl, ...
               'tag_suffix', sfx, 'save_inner_full', false);
    fprintf('\n\n>>>>>> %s  %s  %dx%d  inner_budget=%d\n', series, vars{k}, nelx, nely, ib);
    try
        run_variant(vars{k}, nelx, nely, 'CC', o);
    catch ME
        fprintf(2, 'RUN FAILED %s: %s\n', vars{k}, ME.message);
        fprintf(2, '%s\n', getReport(ME));
    end
end
fprintf('\n>>>>>> SERIES %s COMPLETE\n', series);
end
