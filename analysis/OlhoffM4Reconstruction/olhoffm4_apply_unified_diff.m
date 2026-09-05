function outText = olhoffm4_apply_unified_diff(baseText, diffText)
%OLHOFFM4_APPLY_UNIFIED_DIFF  Apply a unified diff in memory, strictly.
%
%   outText = OLHOFFM4_APPLY_UNIFIED_DIFF(baseText, diffText) applies the
%   unified diff in DIFFTEXT to BASETEXT and returns the result.  Both inputs
%   are raw file contents as char.
%
%   It exists so that a DECLARED modification to the imported solver can be
%   PROVED rather than asserted: hashing
%
%       apply(patches/<x>.diff, patches/<x>.source-verbatim)
%
%   against the manifest's sha256_imported shows that the file which actually
%   runs is the audited source plus exactly the declared diff and nothing else.
%   That proof needs no access to the external source repository, so it stays
%   available for as long as this repository does.
%
%   Deliberately strict.  Every context and deletion line must match the base
%   text exactly; there is no fuzz, no offset search and no partial
%   application.  A diff that does not apply cleanly is a failure, not
%   something to be reconciled.
%
%   Limitations, all of them deliberate: single-file diffs only, no
%   "\ No newline at end of file" markers, hunks in ascending non-overlapping
%   order.  The import tooling produces exactly this form.
%
%   See also OLHOFFM4_VERIFY_IMPORT, OLHOFFM4_SHA256_BYTES.

LF = char(10); %#ok<CHARTEN>

[baseLines, baseEndsNewline] = splitLines(baseText, LF);
diffLines                    = splitLines(diffText, LF);

out = cell(1, numel(baseLines));   % preallocated to the base size; grows if the diff adds
nOut = 0;
src  = 1;                          % next unconsumed base line
i    = 1;

while i <= numel(diffLines)
    L = diffLines{i};
    if numel(L) < 2 || ~strcmp(L(1:2), '@@')
        i = i + 1;                 % ---/+++ headers, and anything before the first hunk
        continue
    end

    % The counts are captured WITH their leading comma: MATLAB's regexp does
    % not return tokens from groups nested inside a non-capturing (?:...)?,
    % so '(?:,(\d+))?' would silently yield two tokens instead of four.
    tok = regexp(L, '^@@\s+-(\d+)(,\d+)?\s+\+(\d+)(,\d+)?\s+@@', 'tokens', 'once');
    if isempty(tok)
        error('olhoffm4_apply_unified_diff:MalformedHunkHeader', ...
            'Malformed hunk header: %s', L);
    end
    oldStart = str2double(tok{1});
    oldCount = hunkCount(tok{2});
    newCount = hunkCount(tok{4});

    if oldStart < src
        error('olhoffm4_apply_unified_diff:HunksOutOfOrder', ...
            'Hunk at base line %d overlaps or precedes already-consumed line %d.', ...
            oldStart, src);
    end
    if oldStart > numel(baseLines) + 1
        error('olhoffm4_apply_unified_diff:HunkPastEnd', ...
            'Hunk starts at base line %d but the base has only %d lines.', ...
            oldStart, numel(baseLines));
    end

    % ---- unchanged region ahead of the hunk ----------------------------
    while src < oldStart
        nOut = nOut + 1; out{nOut} = baseLines{src}; src = src + 1;
    end

    % ---- the hunk body --------------------------------------------------
    i = i + 1;
    usedOld = 0; madeNew = 0;
    while i <= numel(diffLines) && (usedOld < oldCount || madeNew < newCount)
        B = diffLines{i};
        if isempty(B)
            tag = ' '; body = '';          % some tools emit a bare empty context line
        else
            tag = B(1); body = B(2:end);
            if isempty(body); body = ''; end   % 1x0 -> 0x0; strcmp compares size
        end
        switch tag
            case ' '
                assertBaseLine(baseLines, src, body, i);
                nOut = nOut + 1; out{nOut} = baseLines{src};
                src = src + 1; usedOld = usedOld + 1; madeNew = madeNew + 1;
            case '-'
                assertBaseLine(baseLines, src, body, i);
                src = src + 1; usedOld = usedOld + 1;
            case '+'
                nOut = nOut + 1; out{nOut} = body;
                madeNew = madeNew + 1;
            case '\'
                error('olhoffm4_apply_unified_diff:NoNewlineMarkerUnsupported', ...
                    ['Diff line %d carries a "\\ No newline at end of file" marker; ' ...
                     'this applier handles newline-terminated files only.'], i);
            otherwise
                error('olhoffm4_apply_unified_diff:BadHunkLine', ...
                    'Diff line %d has unexpected prefix ''%s'': %s', i, tag, B);
        end
        i = i + 1;
    end
    if usedOld ~= oldCount || madeNew ~= newCount
        error('olhoffm4_apply_unified_diff:TruncatedHunk', ...
            ['Hunk at base line %d is truncated: consumed %d of %d old lines and ' ...
             'produced %d of %d new lines.'], oldStart, usedOld, oldCount, madeNew, newCount);
    end
end

% ---- whatever the last hunk left behind ---------------------------------
while src <= numel(baseLines)
    nOut = nOut + 1; out{nOut} = baseLines{src}; src = src + 1;
end

outText = strjoin(out(1:nOut), LF);
if baseEndsNewline
    outText = [outText LF];
end
end

% =========================================================================
function [lines, endsNewline] = splitLines(txt, LF)
txt = char(reshape(txt, 1, []));
endsNewline = ~isempty(txt) && txt(end) == LF;
if endsNewline; txt = txt(1:end-1); end
if isempty(txt)
    lines = {};
else
    lines = strsplit(txt, LF, 'CollapseDelimiters', false);
end
end

function n = hunkCount(tok)
% TOK is ',<count>' or '' -- an absent count in "@@ -a +c @@" means one line,
% per the unified format.
if isempty(tok)
    n = 1;
else
    n = str2double(tok(2:end));
end
if ~isfinite(n) || n < 0 || mod(n,1) ~= 0
    error('olhoffm4_apply_unified_diff:MalformedHunkHeader', ...
        'Unreadable hunk line count ''%s''.', tok);
end
end

function assertBaseLine(baseLines, src, body, diffLineNo)
if src > numel(baseLines)
    error('olhoffm4_apply_unified_diff:BaseExhausted', ...
        'Diff line %d expects base line %d, but the base has only %d lines.', ...
        diffLineNo, src, numel(baseLines));
end
b = baseLines{src};
% strcmp is false for two empty char arrays of different shape ('' is 0-by-0,
% B(2:end) of a one-character line is 1-by-0), and blank context lines are
% common, so empties are compared as equal explicitly.
if ~((isempty(b) && isempty(body)) || strcmp(b, body))
    error('olhoffm4_apply_unified_diff:ContextMismatch', ...
        'Diff line %d does not match base line %d.\n  diff: <%s>\n  base: <%s>', ...
        diffLineNo, src, body, b);
end
end
