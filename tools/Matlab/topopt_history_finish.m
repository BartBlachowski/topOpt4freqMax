function H = topopt_history_finish(H)
%TOPOPT_HISTORY_FINISH  Trim preallocation and derive k_cont.
%
%   H = TOPOPT_HISTORY_FINISH(H) trims every array to the number of recorded
%   iterations and adds:
%
%     H.k_cont  the iteration at which the last recorded continuation or stage
%               transition occurred, or 0 if the method has none.  Reported
%               beside k* so a reader can see how much of an accepted iteration
%               count is schedule floor and how much is convergence.
%
%   k_cont here is the last observed transition.  A method whose continuation
%   also imposes a minimum number of polish iterations at its final setting has
%   a floor beyond this value; that offset is declared per method in the freeze
%   record (examples/Performance/ledger/freeze_record.json, continuation_floors)
%   rather than guessed from the history.

names = {'iter','stage','stage_iter','d_inf','d_rms','rV','grayness', ...
    'd_inf_design','d_rms_design','objective','elapsed_s', ...
    'move_active_frac','omega1','mode_index','mac'};
for i = 1:numel(names)
    H.(names{i}) = H.(names{i})(1:H.n);
end
H.capacity = H.n;

if isempty(H.markers)
    H.k_cont = 0;
else
    H.k_cont = max([H.markers.iter]);
end
end
