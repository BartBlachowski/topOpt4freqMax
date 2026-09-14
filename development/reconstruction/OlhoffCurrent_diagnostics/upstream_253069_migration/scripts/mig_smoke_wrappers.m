function mig_smoke_wrappers()
%MIG_SMOKE_WRAPPERS  Quick no-solve exercise of the migrated wrapper layer.
P = mig_paths();
restoredefaultpath; addpath(P.scripts); addpath(P.oc);
guard = olhoffcurrent_paths(); %#ok<NASGU>
st = olhoffcurrent_currentness('Verbose', true);
prov = olhoffcurrent_provenance();
fprintf('production preset: %s (event %s)\n', prov.production_preset, prov.production_preset_event_date);
R = olhoffcurrent_presets();
for k = 1:numel(R)
    [cfg, info] = olhoffcurrent_config(160, 20, 'Preset', R(k).name);
    fprintf('%-62s stiff=%-8s mass=%-4s move=%-8s signal=%-15s stop=%-15s levels=%s cap=%d hash=%s\n', ...
        info.name, cfg.material.stiffness.model, cfg.material.mass.model, cfg.move.policy, ...
        cfg.move.continuation.signal, cfg.stop.rule, mat2str(cfg.move.levels), cfg.runtime.maxOuter, ...
        olhoffcurrent_config_hash(cfg));
    s = olhoffcurrent_caveat(R(k).name); assert(numel(s) > 200);
end
[c1, i1] = olhoffcurrent_config(160, 20, 'Preset', 'duOlhoffFixedPenaltySensitivityFiltered');
c0 = olhoffcurrent_config(160, 20, 'Preset', P.name.beta);
fprintf('compat alias -> %s via %s, same hash=%d\n', i1.name, i1.resolvedVia, ...
    strcmp(olhoffcurrent_config_hash(c1), olhoffcurrent_config_hash(c0)));
bad = {'', 'M4', 'S160x20', 'duOlhoffFrozenM4', 'duOlhoffAdaptivePedersen', 'nonsense'};
for k = 1:numel(bad)
    try
        olhoffcurrent_config(160, 20, 'Preset', bad{k}); fprintf('NOT REFUSED: "%s"\n', bad{k});
    catch ME
        fprintf('refused "%s": %s\n', bad{k}, ME.identifier);
    end
end
try, olhoffcurrent_config(160, 20); fprintf('NOT REFUSED: unnamed\n'); catch ME, fprintf('refused unnamed: %s\n', ME.identifier); end
disp(olh.config.describe(olhoffcurrent_config(160, 20, 'Preset', P.name.ex3)));
end
