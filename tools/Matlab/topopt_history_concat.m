function H = topopt_history_concat(varargin)
%TOPOPT_HISTORY_CONCAT  Join per-stage histories into one global history.
%
%   H = TOPOPT_HISTORY_CONCAT(H1, H2, ...) concatenates finished histories in
%   execution order, renumbering the global iteration column so it runs
%   1..sum(n) across stages while leaving stage and stage_iter untouched.
%   Marker iterations are shifted by the same offset.
%
%   Used by staged methods so that Total = Stage 1 + Stage 2 holds in the
%   history exactly as it must in the printed tables.
%
%   Empty or omitted inputs are skipped, so a run that never entered its second
%   stage concatenates cleanly.

names = {'iter','stage','stage_iter','d_inf','d_rms','rV','grayness', ...
    'd_inf_design','d_rms_design','objective','elapsed_s', ...
    'move_active_frac','omega1','mode_index','mac'};

H = [];
offset = 0;
elapsedOffset = 0;

for a = 1:numel(varargin)
    Hi = varargin{a};
    if isempty(Hi) || ~isstruct(Hi) || Hi.n == 0
        continue;
    end
    if isempty(H)
        H = Hi;
        H.iter = (1:H.n)';
        offset = H.n;
        elapsedOffset = localLastFinite(H.elapsed_s);
        continue;
    end

    for i = 1:numel(names)
        if strcmp(names{i}, 'iter')
            H.iter = [H.iter; offset + (1:Hi.n)'];
        elseif strcmp(names{i}, 'elapsed_s')
            % Each stage times itself from zero; make the joined column
            % cumulative across the whole run.
            H.elapsed_s = [H.elapsed_s; elapsedOffset + Hi.elapsed_s];
        else
            H.(names{i}) = [H.(names{i}); Hi.(names{i})];
        end
    end

    for m = 1:numel(Hi.markers)
        mk = Hi.markers(m);
        mk.iter = mk.iter + offset;
        H.markers(end+1) = mk;
    end

    offset = offset + Hi.n;
    elapsedOffset = elapsedOffset + localLastFinite(Hi.elapsed_s);
end

if isempty(H)
    H = topopt_history_init(1, struct());
    H = topopt_history_finish(H);
    return;
end

H.n = offset;
H.capacity = offset;
if isempty(H.markers)
    H.k_cont = 0;
else
    H.k_cont = max([H.markers.iter]);
end
end

function v = localLastFinite(col)
idx = find(isfinite(col), 1, 'last');
if isempty(idx)
    v = 0;
else
    v = col(idx);
end
end
