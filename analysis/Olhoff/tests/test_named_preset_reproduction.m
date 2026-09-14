function nFail = test_named_preset_reproduction(cases)
%TEST_NAMED_PRESET_REPRODUCTION  160x20 bitwise reproduction of a named preset.
%
%   nFail = TEST_NAMED_PRESET_REPRODUCTION('pedersen')         about 4-5 min
%       duOlhoffPedersenAdaptiveBoxSensitivityFiltered, resolved through
%       olhoffcurrent_config exactly as production resolves it, against the
%       COMMITTED upstream sweep result S160x20 (Olhoff 6b08708 / 253069,
%       fixtures/S160x20_reference.json)
%
%   nFail = TEST_NAMED_PRESET_REPRODUCTION('stageExhaustion')  about 5-6 min
%       duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered (diagnostics
%       on, as its historical runs) against the PRE-MIGRATION OlhoffCurrent
%       result of the same configuration (fixtures/EX3_160_reference.json)
%
%   nFail = TEST_NAMED_PRESET_REPRODUCTION()  both.
%
%   The beta-stall preset is covered by TEST_PRESET_EQUIVALENCE against the saved
%   conference campaign.
%
%   STANDARD.  Bitwise, through SHA-256 digests of class, size and raw bytes
%   (OLHOFFCURRENT_TEST_DIGEST): rho, omega, lambda, every non-timing hist field,
%   the log text, status, outer and cumulative inner counts, and res.exhaustion
%   or res.aux where the reference has them.  Timing fields are excluded.  The
%   migrated result may carry fields the reference does not have only where they
%   are declared reporting additions below; anything else fails.
%
%   Scientific reproduction is asserted at 160x20 only.

if nargin < 1 || isempty(cases), cases = {'pedersen', 'stageExhaustion'}; end
if ischar(cases), cases = {cases}; end

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(root); addpath(here);
maxNumCompThreads(1);

nFail = 0;
fprintf('\n%s\nTEST_NAMED_PRESET_REPRODUCTION\n%s\n', repmat('=',1,72), repmat('=',1,72));

for w = 1:numel(cases)
    switch cases{w}
        case 'pedersen'
            preset = 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered';
            fx = 'S160x20_reference.json';
            opts = {};
            allowedExtraTop = {};             % the 6b08708 reference already has res.aux
        case 'stageExhaustion'
            preset = 'duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered';
            fx = 'EX3_160_reference.json';
            opts = {'Diagnostics', true};
            allowedExtraTop = {'aux'};        % res.aux: reporting (Mnd, mean box) added upstream at 6b08708
        otherwise
            error('test_named_preset_reproduction:Unknown', 'unknown case %s', cases{w});
    end
    ref = jsondecode(fileread(fullfile(here, 'fixtures', fx)));

    guard = olhoffcurrent_paths(); %#ok<NASGU>
    cfg = olhoffcurrent_config(160, 20, 'Preset', preset, opts{:});
    res = olhoffSolve(cfg);
    got = olhoffcurrent_test_digest(res);
    clear guard

    fprintf('  --- %s @ 160x20 vs %s ---\n', preset, fx);
    nFail = nFail + chk('design vector rho (bitwise)', strcmp(got.rho, ref.rho));
    nFail = nFail + chk('eigenvalues omega and lambda (bitwise)', ...
        strcmp(got.omega, ref.omega) && strcmp(got.lambda, ref.lambda));
    nFail = nFail + chk(sprintf('outer iterations %d / %d', got.nOuter, ref.nOuter), got.nOuter == ref.nOuter);
    nFail = nFail + chk(sprintf('cumulative inner iterations %d / %d', got.innerTotal, ref.innerTotal), ...
        got.innerTotal == ref.innerTotal);
    nFail = nFail + chk(sprintf('status %s / %s', got.status, ref.status), strcmp(got.status, ref.status));
    nFail = nFail + chk('solver log text', strcmp(got.log, ref.log));

    refHist = fieldnames(ref.hist); bad = {};
    for k = 1:numel(refHist)
        if ~isfield(got.hist, refHist{k}) || ~strcmp(got.hist.(refHist{k}), ref.hist.(refHist{k}))
            bad{end+1} = refHist{k}; %#ok<AGROW>
        end
    end
    nFail = nFail + chk(sprintf('all %d non-timing hist fields of the reference (bitwise)', numel(refHist)), ...
        isempty(bad));
    if ~isempty(bad), fprintf('      differing: %s\n', strjoin(bad, ', ')); end
    extraHist = setdiff(fieldnames(got.hist), refHist);
    nFail = nFail + chk('no hist field beyond the reference (timing excluded)', isempty(extraHist));
    if ~isempty(extraHist), fprintf('      extra: %s\n', strjoin(extraHist, ', ')); end

    if isfield(ref, 'aux')
        a = fieldnames(ref.aux); badA = {};
        for k = 1:numel(a)
            if ~isfield(got, 'aux') || ~isfield(got.aux, a{k}) || ~strcmp(got.aux.(a{k}), ref.aux.(a{k}))
                badA{end+1} = a{k}; %#ok<AGROW>
            end
        end
        nFail = nFail + chk('res.aux (bitwise)', isempty(badA));
    end
    if isfield(ref, 'exhaustion')
        nFail = nFail + chk('res.exhaustion (bitwise)', isfield(got, 'exhaustion') && ...
            strcmp(got.exhaustion, ref.exhaustion));
    end
    extraTop = setdiff(intersect(fieldnames(got), {'aux', 'exhaustion'}), fieldnames(ref));
    nFail = nFail + chk(sprintf('additions limited to declared reporting fields {%s}', ...
        strjoin(allowedExtraTop, ',')), all(ismember(extraTop, allowedExtraTop)));
    fprintf('    omega1 %s / %s\n', got.omega_17g{1}, ref.omega_17g{1});
end
fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

function n = chk(label, ok)
if ok, fprintf('    [PASS] %s\n', label); n = 0;
else,  fprintf('    [FAIL] %s\n', label); n = 1; end
end
