function resolved = olhoffm4_assert_dispatch(names, root)
%OLHOFFM4_ASSERT_DISPATCH  Fail-closed proof of which implementation will run.
%
%   resolved = OLHOFFM4_ASSERT_DISPATCH(names, root) records WHICH.M FILE each
%   name resolves to and refuses to return unless every one of them lies inside
%   ROOT and outside every path named by OLHOFFM4_FORBIDDEN_PATHS.
%
%   resolved is an n-by-1 struct array with fields name and file, suitable for
%   writing straight into a benchmark manifest.  Recording the resolution is
%   half the point: the manifest then says which code produced the numbers,
%   rather than which code was intended to.

if nargin < 2 || isempty(root); root = olhoffm4_root(); end

repoRoot = fileparts(fileparts(root));      % <repo>/analysis/OlhoffM4Reconstruction
forbidden = olhoffm4_forbidden_paths();

resolved = struct('name', {}, 'file', {});
bad = {};
for i = 1:numel(names)
    f = which(names{i});
    resolved(end+1) = struct('name', names{i}, 'file', f); %#ok<AGROW>
    if isempty(f)
        bad{end+1} = sprintf('%s -> UNRESOLVED', names{i}); %#ok<AGROW>
        continue
    end
    if ~strncmp(f, root, numel(root))
        bad{end+1} = sprintf('%s -> %s (outside the import root)', names{i}, f); %#ok<AGROW>
        continue
    end
    for k = 1:numel(forbidden)
        p = fullfile(repoRoot, forbidden{k});
        if strncmp(f, p, numel(p))
            bad{end+1} = sprintf('%s -> %s (SUPERSEDED implementation)', names{i}, f); %#ok<AGROW>
        end
    end
end

if ~isempty(bad)
    error('olhoffm4_assert_dispatch:WrongImplementation', ...
        ['The Du-Olhoff reconstruction did NOT resolve to the imported ' ...
         'conference implementation:\n  %s\n' ...
         'Refusing to run.  See analysis/OLHOFF_IMPLEMENTATION_STATUS.md.'], ...
        strjoin(bad, sprintf('\n  ')));
end
end
