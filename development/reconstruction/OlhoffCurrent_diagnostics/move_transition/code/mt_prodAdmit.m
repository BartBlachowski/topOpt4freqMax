function [stopIter, info] = mt_prodAdmit(P, tolOuter)
%MT_PRODADMIT  Replay the PRODUCTION admission rule on a recorded trajectory.
%
%   Production: stop.norm = 'l2' with stop.guards.settledMove = true, i.e.
%       ||drho||_2 < tolOuter   AND   move(k) == move(k-1)
%   (stop.guards.ladderExhausted and .maxDesignChange are OFF in production, and
%   projection / p-continuation are off, so nothing else can block the break.)
raw = P.l2 < tolOuter;
settled = [false; P.move(2:end) == P.move(1:end-1)];
admit = raw & settled;
stopIter = find(admit, 1);
info = struct('raw',raw,'settled',settled,'admit',admit,'tolOuter',tolOuter);
if isempty(stopIter); info.status = 'CAP_HIT'; else; info.status = 'ADMITTED'; end
end
