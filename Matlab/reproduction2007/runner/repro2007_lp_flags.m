function flags = repro2007_lp_flags(res, n)
%REPRO2007_LP_FLAGS  Per-outer-iteration linprog exit flag of the Eq. (22) route.
%
%   flags = REPRO2007_LP_FLAGS(res)     one flag per recorded outer iteration
%   flags = REPRO2007_LP_FLAGS(res, n)  the first n of them
%
%   INNERLOOPLP records the numeric linprog exit flag in st.lpFlag, but the
%   imported OLHOFFOPT keeps only the boolean st.conv in hist.innerConv and
%   writes the numeric flag into res.log, and only when the solve FAILS:
%
%       log{k} = 'iter %d: LP inner solve failed (flag=%d)'
%
%   Reconstructing the flag therefore needs both records.  Doing it here, once,
%   keeps the reconstruction identical everywhere it is used -- the history
%   table (REPRO2007_HISTORY) and the stop classification (RUN_REPRO2007) must
%   not be able to disagree about whether the subproblem failed.
%
%   Returns NaN for every iteration of an MMA-route run: that route has no
%   linprog exit flag, and a fabricated 1 would read as "LP succeeded".
%
%   This function reads what the frozen implementation already recorded.  It
%   does not modify, wrap or re-run any part of it.
%
%   See also REPRO2007_HISTORY, RUN_REPRO2007, INNERLOOPLP.

if nargin < 2 || isempty(n)
    n = numel(res.hist.N);
end

if ~strcmpi(res.cfg.innerSolver, 'lp')
    flags = NaN(1, n);
    return
end

conv = logical(res.hist.innerConv(:).');
flags = NaN(1, n);
k = min(n, numel(conv));
idx = false(1, n);
idx(1:k) = conv(1:k);
flags(idx) = 1;                  % innerLoopLP sets conv = (flag == 1)

for i = 1:numel(res.log)
    tok = regexp(res.log{i}, ...
        '^iter (\d+): LP inner solve failed \(flag=(-?\d+)\)', 'tokens', 'once');
    if isempty(tok)
        continue
    end
    j = str2double(tok{1});
    if j >= 1 && j <= n
        flags(j) = str2double(tok{2});
    end
end
end
