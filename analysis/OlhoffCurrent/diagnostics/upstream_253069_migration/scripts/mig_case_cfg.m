function cfg = mig_case_cfg(side, caseName, nelx, nely)
%MIG_CASE_CFG  The configuration of one migration case, resolved on one side.
%   side  'pre'  : worktree BEFORE promotion, old OlhoffCurrent API (edbfe47)
%         'post' : worktree AFTER migration, new named-preset API
%         'up'   : read-only 253069 snapshot, olh.config.resolve with upstream names
%   caseName  BETA | EX3 | EX4 | PED
%   All sides give the SAME runtime fields (cap, single thread, diagnostics,
%   verbose, name), so any difference in a result is a code difference.
if nargin < 3, nelx = 160; nely = 20; end
P = mig_paths();
m = {'domain.mesh.nelx', nelx, 'domain.mesh.nely', nely};
ex3 = {'move.levels', [0.04 0.02 0.01], 'move.continuation.signal', 'stageExhaustion', ...
       'stop.rule', 'stageExhaustion'};
ex4 = {'move.continuation.signal', 'stageExhaustion', 'stop.rule', 'stageExhaustion'};
name = sprintf('OLHOFF_CURRENT_%dx%d', nelx, nely);
rt400  = {'runtime.maxOuter', 400,  'runtime.singleThread', true, ...
          'runtime.diagnostics', false, 'runtime.verbose', false, 'runtime.name', name};
rt1600 = {'runtime.maxOuter', 1600, 'runtime.singleThread', true, ...
          'runtime.diagnostics', true,  'runtime.verbose', false, 'runtime.name', name};

switch side
    case 'pre'
        switch caseName
            case 'BETA'
                cfg = olhoffcurrent_config(nelx, nely);
            case 'EX3'
                info = olhoffcurrent_preset();
                cfg = olh.config.resolve(info.upstreamPreset, m{:}, ex3{:}, rt1600{:});
            case 'EX4'
                info = olhoffcurrent_preset();
                cfg = olh.config.resolve(info.upstreamPreset, m{:}, ex4{:}, rt1600{:});
            otherwise
                error('mig:case', 'case %s has no pre-migration realization', caseName);
        end
    case 'post'
        switch caseName
            case 'BETA'
                cfg = olhoffcurrent_config(nelx, nely, 'Preset', P.name.beta);
            case 'EX3'
                cfg = olhoffcurrent_config(nelx, nely, 'Preset', P.name.ex3, 'Diagnostics', true);
            case 'EX4'
                % a documented OVERRIDE of the historical beta-stall preset, not a registered preset
                info = olhoffcurrent_preset(P.name.beta);
                cfg = olh.config.resolve(info.upstreamPreset, m{:}, ex4{:}, rt1600{:});
            case 'PED'
                cfg = olhoffcurrent_config(nelx, nely, 'Preset', P.name.ped);
            otherwise
                error('mig:case', 'case %s', caseName);
        end
    case 'up'
        switch caseName
            case 'BETA', cfg = olh.config.resolve('duOlhoffFrozenM4', m{:}, rt400{:});
            case 'EX3',  cfg = olh.config.resolve('duOlhoffFrozenM4', m{:}, ex3{:}, rt1600{:});
            case 'EX4',  cfg = olh.config.resolve('duOlhoffFrozenM4', m{:}, ex4{:}, rt1600{:});
            case 'PED',  cfg = olh.config.resolve('duOlhoffAdaptivePedersen', m{:}, rt400{:});
            otherwise, error('mig:case', 'case %s', caseName);
        end
    otherwise
        error('mig:side', 'side %s', side);
end
end
