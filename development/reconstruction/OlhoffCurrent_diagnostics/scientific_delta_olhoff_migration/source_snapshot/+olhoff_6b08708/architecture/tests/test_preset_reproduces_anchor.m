function nFail = test_preset_reproduces_anchor()
%TEST_PRESET_REPRODUCES_ANCHOR  Close the loop on the compatibility layer.
%
%   The twelve behavioural anchors drive the LEGACY path:
%       legacy flat cfg -> olhoffOpt shim -> fromLegacy -> olhoffSolve.
%   This test drives the CANONICAL path instead:
%       olh.config.resolve(preset) -> olhoffSolve
%   and requires the same science digest.
%
%   Passing both means the preset layer is not merely config-equivalent to the
%   historical realization (which test_presets_match_history establishes) but
%   TRAJECTORY-equivalent to it.

root = '/Users/piotrek/Programming/Matlab/Olhoff';
addpath(fullfile(root,'architecture','anchors','code'));
nFail = 0;

% preset                anchor whose reference it must reproduce
C = {
'duOlhoffFrozenM4',    'A1_frozen160'
'duOlhoffMatureM4',    'A2_mature160'
'restorationLadderGuard','A3_r1ladder160'
};

for k = 1:size(C,1)
    preset = C{k,1};  label = C{k,2};
    R = load(fullfile(root,'architecture','anchors','reference',[label '.mat']));
    % The anchor's own runtime settings, which are not part of the formulation.
    ws = warning('off','olh:config:suspicious');
    cfg = olh.config.resolve(preset, ...
        'domain.mesh.nelx', R.cfg.nelx, 'domain.mesh.nely', R.cfg.nely, ...
        'runtime.maxOuter', R.cfg.maxOuter, ...
        'runtime.diagnostics', true, 'runtime.verbose', false, ...
        'runtime.name', label);
    warning(ws);

    res = olhoffSolve(cfg);
    rec = anchorRecord(res, struct('name',label,'maxOuter',R.cfg.maxOuter));
    rec.meta = R.rec.meta;
    d   = anchorDigests(rec);
    ref = anchorDigests(R.rec);

    if strcmp(d.science, ref.science)
        fprintf('  ok   preset %-24s reproduces %s BITWISE via the canonical path\n', preset, label);
    else
        fprintf('  FAIL preset %-24s differs from %s\n', preset, label);
        anchorExplain(R.rec, rec);
        nFail = nFail + 1;
    end
end
end
