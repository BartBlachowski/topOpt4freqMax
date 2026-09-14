function sd_assert_resolution(root, names)
%SD_ASSERT_RESOLUTION  Every name must resolve, the winner must be under root,
%   and there must be no other candidate on the path.  Fails closed.
for i = 1:numel(names)
    w = which(names{i}, '-all');
    if ischar(w), w = {w}; end
    assert(~isempty(w), 'sd:path', '%s does not resolve', names{i});
    assert(strncmp(w{1}, root, numel(root)), 'sd:path', '%s resolves to %s, outside %s', names{i}, w{1}, root);
    others = w(~strncmp(w, root, numel(root)));
    % Class methods (@cls/f) and package members (+pkg/f) cannot take part in
    % bare-name resolution; skipped exactly as olhoffcurrent_assert_dispatch does.
    others = others(~contains(others, [filesep '@']) & ~contains(others, [filesep '+']));
    % mmasub/subsolv: useMMA selects mma_published, the mma/ copy is a sibling in the same root
    assert(isempty(others), 'sd:path', '%s has a foreign candidate: %s', names{i}, strjoin(others, ', '));
end
end
