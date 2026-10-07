function n = confbench_frozen_budget(methodKey)
%CONFBENCH_FROZEN_BUDGET  The frozen per-stage safety budget of a method.
%
%   n = CONFBENCH_FROZEN_BUDGET(methodKey) reads max_iters from the profile
%   freeze manifest that CONFBENCH_METHOD_CONFIG reads (Proposed, Yuksel), or
%   the runtime default maxOuter of the benchmark's Olhoff preset, so the
%   number the driver compares against is the frozen number itself and not a
%   copy of it that can drift.
%
%   The manifest records this value's role explicitly:
%
%       "max_iters_role": "per-stage safety budget; CAP_HIT is not convergence"
%
%   which is what makes RAISING it a different kind of act from lowering it.
%   The budget exists to stop a runaway, not to define the answer: raising it
%   lets a method run until its own stopping rule fires, while lowering it
%   truncates the method before that rule can be read.  See the scientific-
%   evidence rule in PERFORMANCE_COMPARISON.
%
%   See also CONFBENCH_METHOD_CONFIG.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
freezePath = fullfile(repo, 'examples', 'Performance', 'benchmark_profile', ...
    'profile_freeze_manifest.json');

switch lower(char(string(methodKey)))
    case 'yuksel';                    field = 'yuksel_practical';
    case {'proposed', 'ourapproach'}; field = 'proposed_practical';
    case 'olhoff'
        % Not in this manifest: the Du-Olhoff budget is frozen by the named
        % preset the benchmark runs (olhoffcurrent_presets.m, runtime default
        % maxOuter), so it is read from there.
        n = double(olhoffcurrent_preset(confbench_olhoff_preset()).runtimeDefaults.maxOuter);
        return
    otherwise
        error('confbench_frozen_budget:UnknownMethod', ...
            '"%s" has no frozen safety budget.', methodKey);
end

freeze = jsondecode(fileread(freezePath));
n = double(freeze.profiles.(field).max_iters);
end
