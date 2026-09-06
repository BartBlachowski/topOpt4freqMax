function cls = confbench_classify(stopping, nIter, nIterStage, caps)
%CONFBENCH_CLASSIFY  The ONE status-precedence decision for dispatched methods.
%
%   cls = CONFBENCH_CLASSIFY(stopping, nIter, nIterStage, caps) returns a
%   struct with fields status, ok and note.
%
%   PRECEDENCE, single-sourced here:
%
%       SOLVER_FAILURE > CAP_HIT > NATIVE_CONVERGED > UNRECOGNIZED_STOP
%
%   caps carries the ACTUAL numeric iteration caps the run received, read from
%   the runtime configuration/telemetry by the caller and never restated as
%   constants.  All fields optional:
%
%       caps.stage1   compared against nIterStage.stage1
%       caps.stage2   compared against nIterStage.stage2
%       caps.total    compared against nIter
%
%   WHY NUMERIC.  telemetry.stopping exposes ONE stop_reason for the whole run.
%   For a two-stage method that reason is Stage 2's, so a Stage-1 cap hit
%   followed by a Stage-2 tolerance stop reads as a clean convergence.  That is
%   exactly how campaign_9mesh recorded Yuksel 720x90 with
%   stage1_iterations = 1000 = its cap as NATIVE_CONVERGED, and how that
%   censored point entered the fitted scaling exponent.  Text is advisory; the
%   counts against the caps are decisive.
%
%   This function performs no solve and is safe to call from tests.
%
%   See also CONFBENCH_RUN_CASE, CONFBENCH_SELFTEST.

if nargin < 4 || isempty(caps); caps = struct(); end
cls = struct('status', 'UNRECOGNIZED_STOP', 'ok', false, 'note', '');

s = stopping;
reasons = {char(string(fieldOrDefault(s, 'stop_reason', 'N/A')))};
% Kept for forward compatibility: no current solver emits per-stage reasons, so
% these never fire today and the numeric test below is what protects the
% classification.
if isfield(s,'stage1_stop_reason'); reasons{end+1} = char(string(s.stage1_stop_reason)); end
if isfield(s,'stage2_stop_reason'); reasons{end+1} = char(string(s.stage2_stop_reason)); end
capText = any(cellfun(@(r) contains(lower(r), 'max_iter'), reasons));

% ---- numeric cap detection against the ACTUAL runtime caps --------------
capNumeric = false; capParts = {};
capFields = {'stage1', 'stage2', 'total'};
capCounts = {stageCount(nIterStage,'stage1'), stageCount(nIterStage,'stage2'), nIter};
for ci = 1:numel(capFields)
    fld = capFields{ci};
    if ~isfield(caps, fld) || isempty(caps.(fld)); continue; end
    capVal = double(caps.(fld)); nVal = double(capCounts{ci});
    if ~isfinite(capVal) || capVal <= 0 || ~isfinite(nVal); continue; end
    if nVal >= capVal
        capNumeric = true;
        capParts{end+1} = sprintf('%s %g >= cap %g', fld, nVal, capVal); %#ok<AGROW>
    end
end
capHit = capText || capNumeric;

% ---- solver failure, highest precedence ---------------------------------
failed = false;
sf = fieldOrDefault(s, 'subproblem_failed', false);
if ~isempty(sf); failed = failed || logical(sf); end
nsf = fieldOrDefault(s, 'n_subproblem_failures', 0);
if ~isempty(nsf) && isfinite(nsf); failed = failed || nsf > 0; end
if isfield(s, 'design_nonfinite') && logical(s.design_nonfinite); failed = true; end

converged = contains(lower(reasons{1}), 'tolerance');

if failed
    cls.status = 'SOLVER_FAILURE';
    cls.note = 'solver reported a failed subproblem or a nonfinite result';
elseif capHit
    cls.status = 'CAP_HIT';
    if capNumeric
        cls.note = sprintf(['iteration cap reached (%s); NOT convergence ' ...
            '(overall stop_reason was "%s")'], strjoin(capParts, ', '), reasons{1});
    else
        cls.note = sprintf('iteration cap reached (%s); NOT convergence', ...
            strjoin(unique(reasons), '|'));
    end
elseif converged
    cls.status = 'NATIVE_CONVERGED';
    cls.note = sprintf('native stop test met (%s)', reasons{1});
    cls.ok = true;
else
    cls.status = 'UNRECOGNIZED_STOP';
    cls.note = sprintf('stop reason "%s" is not in the frozen vocabulary', reasons{1});
end
end

% =========================================================================
function v = stageCount(nIterStage, name)
v = NaN;
if isstruct(nIterStage) && isfield(nIterStage, name) && ~isempty(nIterStage.(name))
    v = double(nIterStage.(name));
end
end

% =========================================================================
function v = fieldOrDefault(s, name, dflt)
v = dflt;
if isstruct(s) && isfield(s, name) && ~isempty(s.(name)); v = s.(name); end
end
