function P = sd_paths()
%SD_PATHS  Absolute locations used by every audit script (no side effects).
P.audit  = fileparts(fileparts(mfilename('fullpath')));
P.repo   = '/Users/piotrek/Programming/topOpt4freqMax';
P.oc     = fullfile(P.repo, 'analysis', 'OlhoffCurrent');
P.impl   = fullfile(P.oc, '+impl');
P.snap   = fullfile(P.audit, 'source_snapshot', '+olhoff_6b08708');
P.eval   = fullfile(P.audit, 'evaluations');
P.fig    = fullfile(P.audit, 'figures');
P.s480   = fullfile(P.snap, 'repro', 'results', 'S480x60', 'res.mat');
P.c480traj  = fullfile(P.oc, 'evidence', 'three_rung_canary_preflight', 'C480x60_three_rung_trajectory.mat');
P.c480state = fullfile(P.oc, 'evidence', 'three_rung_canary_preflight', 'C480x60_three_rung_state.mat');
P.m1dir  = fullfile(P.eval, 'm1_run');
end
