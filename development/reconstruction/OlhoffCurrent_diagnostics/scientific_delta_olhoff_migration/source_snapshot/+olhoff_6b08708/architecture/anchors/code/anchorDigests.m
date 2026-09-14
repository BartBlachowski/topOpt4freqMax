function d = anchorDigests(rec)
%ANCHORDIGESTS  The two digests an anchor comparison uses.
%
%   d.science      SHA-256 over every scientific quantity: densities, design
%                  variables, eigenfrequencies, volume, the complete non-timing
%                  history, the per-iteration design increments, every
%                  continuation/projection/multiplicity transition, the status
%                  and the failure counters.
%                  ---> Phase 18 requires this to be BITWISE IDENTICAL.
%
%   d.presentation SHA-256 over the human-readable log messages only.
%                  Renaming a stopping guard changes these strings without
%                  changing any trajectory, so they are reported SEPARATELY and
%                  are not part of the equality standard.  A change here must
%                  still be explained.
%
%   d.logShape     iteration-independent structure of the log: how many lines,
%                  and the leading token of each.  A refactor may reword a
%                  message but must not add, drop or reorder one.

sci = rmfield(rec, intersect({'log','nLogLines','meta'}, fieldnames(rec)));
d.science      = anchorDigest(sci);
d.presentation = anchorDigest(struct('log',{rec.log}));
d.logShape     = anchorDigest(struct('n',numel(rec.log),'kinds',{local_kinds(rec.log)}));
end

function k = local_kinds(logLines)
%LOCAL_KINDS  Classify each log line by the event it reports, independent of
%   the wording used to report it.
k = cell(numel(logLines),1);
for i = 1:numel(logLines)
    s = logLines{i};
    if     contains(s,'converged at outer iteration'),        k{i} = 'CONVERGED';
    elseif contains(s,'stop blocked'),                        k{i} = 'STOP_BLOCKED_P';
    elseif contains(s,'baseline stop blocked'),               k{i} = 'STOP_BLOCKED_GUARD';
    elseif contains(s,'below eps but the move limit'),        k{i} = 'STOP_BLOCKED_MOVE';
    elseif contains(s,'consumed by projection'),              k{i} = 'PROJECTION_ADVANCE';
    elseif contains(s,'consumed by p continuation'),          k{i} = 'P_ADVANCE';
    elseif contains(s,'is itself multiple'),                  k{i} = 'OMEGA_J_MULTIPLE';
    elseif contains(s,'J may be truncated'),                  k{i} = 'NMAX_REACHED';
    elseif contains(s,'LP inner solve failed'),               k{i} = 'LP_FAIL';
    else,                                                     k{i} = 'OTHER';
    end
end
end
