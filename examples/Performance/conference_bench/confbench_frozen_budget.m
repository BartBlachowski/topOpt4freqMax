function n = confbench_frozen_budget(methodKey)
%CONFBENCH_FROZEN_BUDGET  The frozen per-stage safety budget of a method.
%
%   n = CONFBENCH_FROZEN_BUDGET(methodKey) reads max_iters from the profile
%   freeze manifest that CONFBENCH_METHOD_CONFIG reads, so the number the
%   driver compares against is the frozen number itself and not a copy of it
%   that can drift.
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
freezePath = fullfile(repo, 'analysis', 'three_method_parametric_study', ...
    'results', 'profile_freeze_manifest.json');

switch lower(char(string(methodKey)))
    case 'yuksel';                    field = 'yuksel_practical';
    case {'proposed', 'ourapproach'}; field = 'proposed_practical';
    otherwise
        error('confbench_frozen_budget:UnknownMethod', ...
            ['"%s" has no per-stage budget in the profile freeze manifest. ' ...
             'The Du-Olhoff reconstruction is frozen by olhoffm4_config.m ' ...
             '(maxOuter), not by this manifest.'], methodKey);
end

freeze = jsondecode(fileread(freezePath));
n = double(freeze.profiles.(field).max_iters);
end
