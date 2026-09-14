function rec = anchorRecord(res, cfg)
%ANCHORRECORD  The regression-relevant content of one solver run.
%
%   Everything here is deterministic: wall-clock and per-phase timings are
%   deliberately EXCLUDED, because they are the only fields that legitimately
%   differ between two runs of identical mathematics.  Phase 18 of the refactor
%   brief requires bitwise equality of everything else.

rec = struct();
rec.name        = cfg.name;

% --- terminal design state ------------------------------------------------
rec.rho         = res.rho(:);                 % the field the FE model used
rec.omega       = res.omega(:);
rec.lambda      = res.lambda(:);
rec.volume      = mean(res.rho);
rec.nOuter      = res.nOuter;

% --- design-variable vs physical-density distinction ----------------------
% Under projection the design variable z and the filtered field are separate
% objects and must both be pinned; without projection they do not exist.
if isfield(res,'z'),       rec.z       = res.z(:);       else, rec.z = [];       end
if isfield(res,'zTilde'),  rec.zTilde  = res.zTilde(:);  else, rec.zTilde = [];  end
if isfield(res,'rhoPhys'), rec.rhoPhys = res.rhoPhys(:); else, rec.rhoPhys = []; end
if isfield(res,'projFinalBeta'), rec.projFinalBeta = res.projFinalBeta;
else,                           rec.projFinalBeta = []; end

% --- complete non-timing history -----------------------------------------
TIMING = {'tEig','tGrad','tInner','tOuter'};
hf = setdiff(fieldnames(res.hist), TIMING);
rec.histFields = sort(hf);
for k = 1:numel(rec.histFields)
    rec.hist.(rec.histFields{k}) = res.hist.(rec.histFields{k});
end

% --- per-iteration design increments (the finest trajectory evidence) ------
if isfield(res,'diag') && isfield(res.diag,'drho')
    rec.drho = res.diag.drho;
else
    rec.drho = {};
end

% --- continuation transitions --------------------------------------------
h = res.hist;
rec.transitions = struct( ...
    'moveStages',   local_changes(h.stage), ...
    'moveLevels',   local_changes(h.move), ...
    'pStages',      local_changes(h.pStage), ...
    'pEvents',      find(h.pEvent(:).' ~= 0), ...
    'massLowSpans', local_changes(h.massLow), ...
    'projStages',   local_changes(h.projStage), ...
    'projEvents',   find(h.projEvent(:).' ~= 0), ...
    'multiplicity', local_changes(h.N));

% --- status and failure counters -----------------------------------------
converged = any(contains(res.log, 'converged at outer iteration'));
if converged
    rec.status = 'CONVERGED';
elseif res.nOuter >= cfg.maxOuter
    rec.status = 'CAP_HIT';
else
    rec.status = 'STOPPED_OTHER';
end
rec.innerFailures  = sum(~res.hist.innerConv);
rec.degenTotal     = sum(res.hist.degen);
rec.multJEvents    = sum(res.hist.multJ);
rec.innerTotal     = res.hist.cumInner(end);
rec.nLogLines      = numel(res.log);
rec.log            = res.log(:);
end

function idx = local_changes(v)
%LOCAL_CHANGES  Iterations at which a piecewise-constant series changes value.
v = double(v(:)).';
if isempty(v), idx = []; return; end
idx = find(diff(v) ~= 0) + 1;
end
